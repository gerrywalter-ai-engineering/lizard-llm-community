#!/usr/bin/env node
// One-step setup for Lizard LLM Community Edition.
//
//   node scripts/setup.mjs                    use prebuilt engines (download if missing)
//   node scripts/setup.mjs --build-from-source  compile both engines locally
//
// Prebuilt engines come from this repository's GitHub release for the
// package.json version, and are checked against the release's SHA256SUMS
// before anything is unpacked. No npm packages are needed at runtime.

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import process from "node:process";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const pkg = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8"));
const REPO = "gerrywalter-ai-engineering/lizard-llm-community";
const ASSET = `lizard-engines-windows-x64-v${pkg.version}.zip`;

const ENGINE_FILES = [
  "native/lizard-native-runner/build/lizard-native-runner.exe",
  "native/lizard-native-runner/build/lizard-native-runner-avx2.exe",
  "native/lizard-native-runner/build/lizard-native-runner-avx512.exe",
  "native/caterpillar-runner/build/caterpillar-runner.exe",
];

const args = new Set(process.argv.slice(2));
const say = (line = "") => console.log(line);

function enginesPresent() {
  return ENGINE_FILES.every((file) => fs.existsSync(path.join(root, file)));
}

function checkNode() {
  const major = Number(process.versions.node.split(".")[0]);
  if (major < 22) {
    throw new Error(`Node.js 22 or newer is required (found ${process.versions.node}). Get it from https://nodejs.org`);
  }
  if (process.platform !== "win32") {
    throw new Error("The native engines use Direct3D 12 and run on Windows 10/11 (x64) only.");
  }
}

async function download(url) {
  const response = await fetch(url, { headers: { "User-Agent": "lizard-llm-community-setup" }, redirect: "follow" });
  if (!response.ok) {
    const error = new Error(`HTTP ${response.status} for ${url}`);
    error.status = response.status;
    throw error;
  }
  return Buffer.from(await response.arrayBuffer());
}

async function installPrebuilt() {
  const base = `https://github.com/${REPO}/releases/download/v${pkg.version}`;
  say(`Downloading prebuilt engines (${ASSET})...`);
  const [zip, sums] = await Promise.all([download(`${base}/${ASSET}`), download(`${base}/SHA256SUMS`)]);
  const expected = sums.toString("utf8").split(/\r?\n/)
    .map((line) => line.trim().split(/\s+\*?/))
    .find(([, name]) => name === ASSET)?.[0];
  const actual = crypto.createHash("sha256").update(zip).digest("hex");
  if (!expected || expected.toLowerCase() !== actual) {
    throw new Error(`Checksum mismatch for ${ASSET}: expected ${expected ?? "(missing)"}, got ${actual}. Nothing was installed.`);
  }
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "lizard-engines-"));
  const zipPath = path.join(tmp, ASSET);
  fs.writeFileSync(zipPath, zip);
  // tar.exe ships with Windows 10 1803+ and reads zip archives.
  const result = spawnSync(windowsTar(), ["-xf", zipPath, "-C", root], { stdio: "inherit" });
  fs.rmSync(tmp, { recursive: true, force: true });
  if (result.status !== 0) throw new Error("Could not unpack the engine archive.");
  say("Checksum verified, engines unpacked.");
}

function run(command, commandArgs, env = process.env) {
  const result = spawnSync(command, commandArgs, { cwd: root, stdio: "inherit", env, shell: false });
  if (result.status !== 0) throw new Error(`${command} ${commandArgs.join(" ")} failed (exit ${result.status}).`);
}

function buildFromSource() {
  const probe = spawnSync("cmake", ["--version"], { stdio: "ignore" });
  if (probe.status !== 0) {
    throw new Error("Building from source needs CMake, Ninja and a C++20 compiler. The easiest way:\n  winget install BrechtSanders.WinLibs.POSIX.UCRT\nthen open a new terminal and run setup again.");
  }
  // Two parallel compile jobs keep the machine usable while it builds.
  const env = { ...process.env, CMAKE_BUILD_PARALLEL_LEVEL: process.env.CMAKE_BUILD_PARALLEL_LEVEL ?? "2" };
  say("Building lizard-native (baseline, AVX2, AVX-512). This takes a few minutes...");
  run(process.execPath, [path.join("scripts", "build-native-runner.mjs")], env);
  say("Building caterpillar...");
  run("cmake", ["-S", "native/caterpillar-runner", "-B", "native/caterpillar-runner/build", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"], env);
  run("cmake", ["--build", "native/caterpillar-runner/build"], env);
}

async function main() {
  say(`Lizard LLM Community Edition ${pkg.version} setup`);
  checkNode();
  if (args.has("--build-from-source")) {
    buildFromSource();
  } else if (enginesPresent() && !args.has("--force")) {
    say("Engines are already installed.");
  } else {
    try {
      await installPrebuilt();
    } catch (error) {
      say(`Prebuilt engines are not available: ${error.message}`);
      say("Falling back to building from source.");
      buildFromSource();
    }
  }
  const missing = ENGINE_FILES.filter((file) => !fs.existsSync(path.join(root, file)));
  if (missing.length) {
    throw new Error(`Setup finished but these engine files are missing:\n  ${missing.join("\n  ")}`);
  }
  say("");
  say("Ready. Start the app with:  start.cmd   (or: npm start)");
  say("Or use the CLI:             node src/cli.js pull Qwen/Qwen2.5-0.5B-Instruct-GGUF");
}

main().catch((error) => {
  console.error(`\nSetup failed: ${error.message}`);
  process.exitCode = 1;
});

// Windows ships bsdtar as System32\tar.exe (it reads and writes zip). Prefer it
// over a GNU tar that Git Bash may put first on PATH, which treats "E:" as a host.
function windowsTar() {
  const systemTar = path.join(process.env.SystemRoot ?? "C:/Windows", "System32", "tar.exe");
  return fs.existsSync(systemTar) ? systemTar : "tar";
}
