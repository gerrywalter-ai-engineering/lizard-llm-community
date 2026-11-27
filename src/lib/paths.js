import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export function expandHome(value) {
  if (!value) {
    return value;
  }
  if (value === "~") {
    return os.homedir();
  }
  if (value.startsWith("~/") || value.startsWith("~\\")) {
    return path.join(os.homedir(), value.slice(2));
  }
  return value;
}

export function ensureDirectory(directoryPath) {
  fs.mkdirSync(directoryPath, { recursive: true });
}

export function sanitizeSlug(input) {
  return String(input)
    .toLowerCase()
    .replace(/[^a-z0-9._:-]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
}

export function sameVolume(left, right) {
  return path.parse(path.resolve(left)).root.toLowerCase() === path.parse(path.resolve(right)).root.toLowerCase();
}
