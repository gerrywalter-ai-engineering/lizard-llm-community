import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const moduleDirectory = path.dirname(fileURLToPath(import.meta.url));

function findPackageRoot(startDirectory) {
  let current = path.resolve(startDirectory);
  while (true) {
    if (fs.existsSync(path.join(current, "package.json"))) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) {
      return null;
    }
    current = parent;
  }
}

export function getAppRoot() {
  if (process.env.LIZARD_APP_ROOT) {
    return path.resolve(process.env.LIZARD_APP_ROOT);
  }
  return findPackageRoot(moduleDirectory) ?? process.cwd();
}

export function resolveAppPath(...segments) {
  return path.join(getAppRoot(), ...segments);
}

export function resolveAppRelativePath(value) {
  if (value == null || value === "") {
    return value;
  }
  return path.isAbsolute(value) ? value : resolveAppPath(value);
}
