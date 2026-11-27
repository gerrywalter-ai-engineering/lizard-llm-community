import fs from "node:fs";
import path from "node:path";
import { ensureDirectory, sanitizeSlug } from "./paths.js";

function timestamp() {
  return new Date().toISOString();
}

function buildLogPath(config) {
  ensureDirectory(config.logging.directory);
  const day = new Date().toISOString().slice(0, 10);
  return path.join(config.logging.directory, `${config.appName}-${day}.log`);
}

export function createCommandLogger({ command, options, config }) {
  const logPath = buildLogPath(config);
  const commandId = `${sanitizeSlug(command)}-${Date.now()}`;

  function write(level, event, data) {
    const record = {
      ts: timestamp(),
      level,
      command,
      commandId,
      event,
      data,
    };
    fs.appendFileSync(logPath, `${JSON.stringify(record)}\n`);
  }

  write("info", "logger-open", { options });

  return {
    path: logPath,
    info(event, data = {}) {
      write("info", event, data);
    },
    warn(event, data = {}) {
      write("warn", event, data);
    },
    error(event, data = {}) {
      write("error", event, data);
    },
    close() {
      write("info", "logger-close", {});
    },
  };
}
