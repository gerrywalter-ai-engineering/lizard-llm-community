#!/usr/bin/env node
// Packs the built engines into dist/lizard-engines-windows-x64-v<version>.zip
// and writes dist/SHA256SUMS, ready to attach to a GitHub release.

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const pkg = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8"));
const asset = `lizard-engines-windows-x64-v${pkg.version}.zip`;
const files = [
  "native/lizard-native-runner/build/lizard-native-runner.exe",
  "native/lizard-native-runner/build/lizard-native-runner-avx2.exe",
  "native/lizard-native-runner/build/lizard-native-runner-avx512.exe",
  "native/caterpillar-runner/build/caterpillar-runner.exe",
];

const missing = files.filter((file) => !fs.existsSync(path.join(root, file)));
if (missing.length) {
  console.error(`Build the engines first (npm run build:engines). Missing:\n  ${missing.join("\n  ")}`);
  process.exit(1);
}
const dist = path.join(root, "dist");
fs.mkdirSync(dist, { recursive: true });
const zipPath = path.join(dist, asset);
fs.rmSync(zipPath, { force: true });
// bsdtar (tar.exe on Windows) writes a zip when the name ends in .zip with -a.
const result = spawnSync("tar", ["-a", "-cf", zipPath, ...files], { cwd: root, stdio: "inherit" });
if (result.status !== 0) process.exit(result.status ?? 1);
const hash = crypto.createHash("sha256").update(fs.readFileSync(zipPath)).digest("hex");
fs.writeFileSync(path.join(dist, "SHA256SUMS"), `${hash}  ${asset}\n`);
console.log(`${asset}\n${hash}`);
