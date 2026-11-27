// Installed MCP plugins: what the user added, whether it is connected, and the
// tools it exposes to the bound model.
//
// A plugin is "added" the moment its command is recorded; it is only spawned
// when a chat turn actually needs its tools, and it is torn down when the
// server shuts down. Connections are cached per plugin id so a conversation
// does not pay the npx cold-start on every message.

import fs from "node:fs";
import path from "node:path";
import { ensureDirectory } from "../paths.js";
import { newId } from "./chat-store.js";
import { McpStdioClient } from "./mcp-client.js";
import { findCatalogEntry, resolveCatalogArgs, MCP_CATALOG } from "./mcp-catalog.js";

function pluginsPath(config) {
  return path.join(config?.runtimeDirectory ?? process.cwd(), "chat", "plugins.json");
}

function readPlugins(config) {
  try {
    const filePath = pluginsPath(config);
    if (!fs.existsSync(filePath)) {
      return [];
    }
    const parsed = JSON.parse(fs.readFileSync(filePath, "utf8"));
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writePlugins(config, plugins) {
  const filePath = pluginsPath(config);
  ensureDirectory(path.dirname(filePath));
  fs.writeFileSync(filePath, JSON.stringify(plugins, null, 2));
  return plugins;
}

// Live clients, keyed by plugin id. Module-scoped so every request handler in
// one server process shares the same connections.
const clients = new Map();

export function listPlugins(config) {
  return readPlugins(config).map((plugin) => ({
    ...plugin,
    status: clients.get(plugin.id)?.status ?? "stopped",
    toolCount: clients.get(plugin.id)?.tools?.length ?? plugin.toolCount ?? 0,
    lastError: clients.get(plugin.id)?.lastError ?? plugin.lastError ?? null,
  }));
}

export function getPlugin(config, pluginId) {
  return readPlugins(config).find((plugin) => plugin.id === pluginId) ?? null;
}

/** Adds a catalog entry, or a fully custom command when catalogId is absent. */
export function addPlugin(config, { catalogId, values = {}, name, command, args = [], env = {} }) {
  const plugins = readPlugins(config);
  let plugin;
  if (catalogId) {
    const entry = findCatalogEntry(catalogId);
    if (!entry) {
      throw new Error(`Unknown plugin "${catalogId}".`);
    }
    if (plugins.some((existing) => existing.catalogId === catalogId && sameValues(existing.values, values))) {
      throw new Error(`${entry.name} is already installed with those settings.`);
    }
    plugin = {
      id: newId("p-"),
      catalogId,
      name: entry.name,
      summary: entry.summary,
      requires: entry.requires,
      command: entry.command,
      args: resolveCatalogArgs(entry, values),
      env: {},
      values,
      enabled: true,
      addedAt: new Date().toISOString(),
    };
  } else {
    if (!command) {
      throw new Error("A custom plugin needs a command to run.");
    }
    plugin = {
      id: newId("p-"),
      catalogId: null,
      name: String(name ?? command).slice(0, 60),
      summary: "Custom MCP server.",
      requires: "custom",
      command: String(command),
      args: Array.isArray(args) ? args.map(String) : [],
      env: env && typeof env === "object" ? env : {},
      values: {},
      enabled: true,
      addedAt: new Date().toISOString(),
    };
  }
  plugins.push(plugin);
  writePlugins(config, plugins);
  return plugin;
}

function sameValues(left = {}, right = {}) {
  const keys = new Set([...Object.keys(left ?? {}), ...Object.keys(right ?? {})]);
  for (const key of keys) {
    if (String(left?.[key] ?? "") !== String(right?.[key] ?? "")) {
      return false;
    }
  }
  return true;
}

export function setPluginEnabled(config, pluginId, enabled) {
  const plugins = readPlugins(config);
  const plugin = plugins.find((entry) => entry.id === pluginId);
  if (!plugin) {
    throw new Error("Plugin not found.");
  }
  plugin.enabled = Boolean(enabled);
  writePlugins(config, plugins);
  if (!plugin.enabled) {
    disconnectPlugin(pluginId).catch(() => {});
  }
  return plugin;
}

export async function removePlugin(config, pluginId) {
  await disconnectPlugin(pluginId);
  writePlugins(config, readPlugins(config).filter((plugin) => plugin.id !== pluginId));
  return { ok: true };
}

export async function connectPlugin(config, pluginId) {
  const existing = clients.get(pluginId);
  if (existing?.status === "ready") {
    return existing;
  }
  const plugin = getPlugin(config, pluginId);
  if (!plugin) {
    throw new Error("Plugin not found.");
  }
  const client = new McpStdioClient({
    id: plugin.id,
    name: plugin.name,
    command: plugin.command,
    args: plugin.args,
    env: plugin.env,
  });
  clients.set(pluginId, client);
  try {
    await client.start();
    await client.listTools();
  } catch (error) {
    clients.delete(pluginId);
    await client.stop().catch(() => {});
    // Record the failure so the plugin card can explain itself after a reload.
    const plugins = readPlugins(config);
    const stored = plugins.find((entry) => entry.id === pluginId);
    if (stored) {
      stored.lastError = error.message;
      writePlugins(config, plugins);
    }
    throw new Error(startupHint(plugin, error));
  }
  const plugins = readPlugins(config);
  const stored = plugins.find((entry) => entry.id === pluginId);
  if (stored) {
    stored.toolCount = client.tools.length;
    stored.lastError = null;
    writePlugins(config, plugins);
  }
  return client;
}

// Turns a spawn failure into something the user can act on. The two ways this
// realistically fails are "the toolchain is missing" and "the package name is
// wrong", and they need different fixes.
function startupHint(plugin, error) {
  const message = error.message ?? String(error);
  if (/ENOENT|not recognized|command not found/i.test(message)) {
    if (plugin.requires === "python") {
      return `${plugin.name} needs uv (uvx) on your PATH. Install it from https://docs.astral.sh/uv, then try again.`;
    }
    if (plugin.requires === "node") {
      return `${plugin.name} needs Node.js (npx) on your PATH. Install Node 20 or newer, then try again.`;
    }
    return `${plugin.name} could not start: "${plugin.command}" was not found on your PATH.`;
  }
  return `${plugin.name} could not start: ${message}`;
}

export async function disconnectPlugin(pluginId) {
  const client = clients.get(pluginId);
  if (!client) {
    return;
  }
  clients.delete(pluginId);
  await client.stop().catch(() => {});
}

export async function disconnectAllPlugins() {
  await Promise.all([...clients.keys()].map((id) => disconnectPlugin(id)));
}

/**
 * Tools the given plugins expose, namespaced so two servers can both ship a
 * "search" without colliding. Connection failures are reported, not thrown:
 * one broken plugin must not take a chat turn down with it.
 */
export async function collectTools(config, pluginIds) {
  const tools = [];
  const failures = [];
  for (const pluginId of pluginIds) {
    const plugin = getPlugin(config, pluginId);
    if (!plugin || !plugin.enabled) continue;
    try {
      const client = await connectPlugin(config, pluginId);
      for (const tool of client.tools) {
        tools.push({
          pluginId,
          pluginName: plugin.name,
          name: `${slug(plugin.name)}.${tool.name}`,
          rawName: tool.name,
          description: tool.description ?? "",
          inputSchema: tool.inputSchema ?? { type: "object", properties: {} },
        });
      }
    } catch (error) {
      failures.push({ pluginId, name: plugin.name, message: error.message });
    }
  }
  return { tools, failures };
}

export function slug(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
}

export async function callTool(config, tools, qualifiedName, args) {
  const tool = tools.find((entry) => entry.name === qualifiedName)
    ?? tools.find((entry) => entry.rawName === qualifiedName);
  if (!tool) {
    return { isError: true, text: `No tool named "${qualifiedName}" is available in this workspace.` };
  }
  const client = await connectPlugin(config, tool.pluginId);
  return client.callTool(tool.rawName, args);
}

/** Catalog plus install state, which is what the marketplace view renders. */
export function marketplaceView(config) {
  const installed = listPlugins(config);
  return MCP_CATALOG.map((entry) => ({
    ...entry,
    installedCount: installed.filter((plugin) => plugin.catalogId === entry.id).length,
  }));
}
