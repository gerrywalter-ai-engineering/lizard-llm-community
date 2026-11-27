// The plugin marketplace: a curated shelf of free, open-source MCP servers.
//
// It ships as data, not as a network call, for the same reason the rest of
// Lizard does: the app has to be usable with the network off. Every entry is a
// published package run on demand by its own package manager (npx for Node,
// uvx for Python), so nothing here is bundled or vendored, and nothing is
// installed until the user adds it.
//
// `requires` names the toolchain the entry needs. The UI shows it up front so
// "add" never fails with an unexplained ENOENT.

export const MCP_CATEGORIES = Object.freeze([
  { id: "files", label: "Files & data" },
  { id: "web", label: "Web" },
  { id: "dev", label: "Developer" },
  { id: "thinking", label: "Reasoning" },
]);

export const MCP_CATALOG = Object.freeze([
  {
    id: "filesystem",
    name: "Filesystem",
    publisher: "Model Context Protocol",
    category: "files",
    license: "MIT",
    requires: "node",
    summary: "Read, search and write files inside folders you choose.",
    description:
      "Gives the model scoped access to directories you name — nothing outside them is reachable. " +
      "Useful for asking questions about a project, summarising a folder of notes, or drafting files in place.",
    homepage: "https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem",
    command: "npx",
    args: ["-y", "@modelcontextprotocol/server-filesystem", "{{directory}}"],
    fields: [
      {
        key: "directory",
        label: "Folder the model may read",
        placeholder: "C:\\Users\\you\\Documents\\notes",
        required: true,
        hint: "Pick the narrowest folder that still answers your questions.",
      },
    ],
  },
  {
    id: "memory",
    name: "Memory",
    publisher: "Model Context Protocol",
    category: "thinking",
    license: "MIT",
    requires: "node",
    summary: "A knowledge graph the model can write to and recall across chats.",
    description:
      "Stores entities, relations and observations the model chooses to remember, and lets it look them up later. " +
      "Facts survive between conversations; the graph stays on this machine.",
    homepage: "https://github.com/modelcontextprotocol/servers/tree/main/src/memory",
    command: "npx",
    args: ["-y", "@modelcontextprotocol/server-memory"],
    fields: [],
  },
  {
    id: "sequential-thinking",
    name: "Sequential Thinking",
    publisher: "Model Context Protocol",
    category: "thinking",
    license: "MIT",
    requires: "node",
    summary: "Structured step-by-step reasoning the model can revise as it goes.",
    description:
      "Gives the model a scratchpad with explicit steps it can branch and revisit. " +
      "Pays off on multi-constraint problems where a single pass tends to skip a step.",
    homepage: "https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking",
    command: "npx",
    args: ["-y", "@modelcontextprotocol/server-sequential-thinking"],
    fields: [],
  },
  {
    id: "everything",
    name: "Everything (reference server)",
    publisher: "Model Context Protocol",
    category: "dev",
    license: "MIT",
    requires: "node",
    summary: "The protocol's own test server — every feature, in one place.",
    description:
      "Exercises prompts, tools, resources and sampling. Add it when you want to confirm that tool calling " +
      "works end to end on your machine before trusting a real server with real data.",
    homepage: "https://github.com/modelcontextprotocol/servers/tree/main/src/everything",
    command: "npx",
    args: ["-y", "@modelcontextprotocol/server-everything"],
    fields: [],
  },
  {
    id: "fetch",
    name: "Fetch",
    publisher: "Model Context Protocol",
    category: "web",
    license: "MIT",
    requires: "python",
    summary: "Pull a web page and hand the model clean Markdown.",
    description:
      "Fetches a URL on request and converts it to Markdown so the model reads content, not tag soup. " +
      "This is the one plugin that reaches the internet — it only does so for URLs you or the model name.",
    homepage: "https://github.com/modelcontextprotocol/servers/tree/main/src/fetch",
    command: "uvx",
    args: ["mcp-server-fetch"],
    fields: [],
  },
  {
    id: "git",
    name: "Git",
    publisher: "Model Context Protocol",
    category: "dev",
    license: "MIT",
    requires: "python",
    summary: "Read history, diffs and branches of a local repository.",
    description:
      "Lets the model answer questions about a repo — what changed, when, and by whom — without you pasting logs.",
    homepage: "https://github.com/modelcontextprotocol/servers/tree/main/src/git",
    command: "uvx",
    args: ["mcp-server-git", "--repository", "{{repository}}"],
    fields: [
      {
        key: "repository",
        label: "Repository folder",
        placeholder: "E:\\projects\\my-app",
        required: true,
      },
    ],
  },
  {
    id: "time",
    name: "Time",
    publisher: "Model Context Protocol",
    category: "dev",
    license: "MIT",
    requires: "python",
    summary: "Current time and timezone conversion.",
    description:
      "Local models have no clock. This gives them one, plus timezone maths that is right instead of plausible.",
    homepage: "https://github.com/modelcontextprotocol/servers/tree/main/src/time",
    command: "uvx",
    args: ["mcp-server-time"],
    fields: [],
  },
  {
    id: "sqlite",
    name: "SQLite",
    publisher: "Model Context Protocol",
    category: "files",
    license: "MIT",
    requires: "python",
    summary: "Query a SQLite database in plain language.",
    description:
      "Points the model at one .sqlite file so it can inspect the schema and run read queries against it.",
    homepage: "https://github.com/modelcontextprotocol/servers/tree/main/src/sqlite",
    command: "uvx",
    args: ["mcp-server-sqlite", "--db-path", "{{database}}"],
    fields: [
      {
        key: "database",
        label: "Database file",
        placeholder: "C:\\data\\app.sqlite",
        required: true,
      },
    ],
  },
]);

export function findCatalogEntry(id) {
  return MCP_CATALOG.find((entry) => entry.id === id) ?? null;
}

/** Substitutes {{field}} placeholders in a catalog command with user values. */
export function resolveCatalogArgs(entry, values = {}) {
  const missing = (entry.fields ?? [])
    .filter((field) => field.required && !String(values[field.key] ?? "").trim())
    .map((field) => field.label);
  if (missing.length > 0) {
    throw new Error(`${entry.name} needs ${missing.join(" and ")} before it can start.`);
  }
  return entry.args.map((arg) =>
    arg.replace(/\{\{(\w+)\}\}/g, (_match, key) => String(values[key] ?? "").trim()),
  );
}
