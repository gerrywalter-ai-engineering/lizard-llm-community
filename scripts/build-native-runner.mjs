import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

export const LIZARD_GGML_COMMIT = "ddd4ec1428a6201e18975ea52b07c71e0f9aef26";
const LIZARD_GGML_REPOSITORY = "https://github.com/ggml-org/llama.cpp.git";

export function parseArgs(argv) {
  const options = {
    buildDir: path.resolve(process.cwd(), "native", "lizard-native-runner", "build"),
    sourceDir: path.resolve(process.cwd(), "native", "lizard-native-runner"),
    clean: false,
    generator: null,
    optimizeForHost: false,
    variants: ["baseline", "avx2", "avx512"],
    toolchainBin: process.env.LIZARD_NATIVE_TOOLCHAIN_BIN
      ? path.resolve(process.env.LIZARD_NATIVE_TOOLCHAIN_BIN)
      : null,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--clean") {
      options.clean = true;
      continue;
    }
    if (arg === "--build-dir" && argv[index + 1]) {
      options.buildDir = path.resolve(argv[index + 1]);
      index += 1;
      continue;
    }
    if (arg === "--source-dir" && argv[index + 1]) {
      options.sourceDir = path.resolve(argv[index + 1]);
      index += 1;
      continue;
    }
    if (arg === "--generator" && argv[index + 1]) {
      options.generator = argv[index + 1];
      index += 1;
      continue;
    }
    if (arg === "--optimize-for-host") {
      options.optimizeForHost = true;
      options.variants = ["host"];
      continue;
    }
    if (arg === "--variants" && argv[index + 1]) {
      options.variants = argv[index + 1].split(",").map((value) => value.trim()).filter(Boolean);
      const invalid = options.variants.filter((value) => !["baseline", "avx2", "avx512"].includes(value));
      if (invalid.length > 0 || options.variants.length === 0) {
        throw new Error(`Unsupported native CPU variant: ${invalid[0] ?? "empty list"}`);
      }
      options.optimizeForHost = false;
      index += 1;
      continue;
    }
    if (arg === "--toolchain-bin" && argv[index + 1]) {
      options.toolchainBin = path.resolve(argv[index + 1]);
      index += 1;
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      console.log(`Usage: node ./scripts/build-native-runner.mjs [options]

Options:
  --clean                   remove the existing build directory before configuring
  --build-dir PATH          override the build directory
  --source-dir PATH         override the source directory
  --generator NAME          override the CMake generator
  --optimize-for-host       enable build-PC CPU instructions (development only;
                            never use for installers or release artifacts)
  --variants LIST           release variants to build (default: baseline,avx2,avx512)
  --toolchain-bin PATH      directory containing cmake.exe, g++.exe, and ninja.exe
  --help                    show this help text

Environment:
  LIZARD_NATIVE_TOOLCHAIN_BIN  preferred toolchain bin directory
  LIZARD_GGML_SOURCE_DIR       optional pinned llama.cpp checkout for offline builds
`);
      process.exit(0);
    }
    throw new Error(`Unknown argument: ${arg}`);
  }

  return options;
}

function candidateToolchainBins() {
  const candidates = [];
  if (process.env.LIZARD_NATIVE_TOOLCHAIN_BIN) {
    candidates.push(path.resolve(process.env.LIZARD_NATIVE_TOOLCHAIN_BIN));
  }
  if (process.platform === "win32") {
    const home = os.homedir();
    candidates.push(
      path.join(home, "AppData", "Local", "Microsoft", "WinGet", "Packages", "BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe", "mingw64", "bin"),
      path.join(home, "AppData", "Local", "Microsoft", "WinGet", "Packages", "BrechtSanders.WinLibs.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe", "mingw64", "bin"),
      "C:\\WinLibs\\mingw64\\bin",
      "C:\\mingw64\\bin",
      "C:\\msys64\\mingw64\\bin",
    );
  }
  return [...new Set(candidates.filter(Boolean))];
}

function resolveFromPath(command) {
  const locator = process.platform === "win32" ? "where.exe" : "which";
  const result = spawnSync(locator, [command], { encoding: "utf8" });
  if (result.status !== 0) {
    return null;
  }
  const first = result.stdout.split(/\r?\n/).map((line) => line.trim()).find(Boolean);
  return first ? path.resolve(first) : null;
}

function resolveExecutable(name, toolchainBin) {
  if (toolchainBin) {
    const direct = path.join(toolchainBin, process.platform === "win32" ? `${name}.exe` : name);
    if (fs.existsSync(direct)) {
      return direct;
    }
  }
  return resolveFromPath(process.platform === "win32" ? `${name}.exe` : name);
}

function detectToolchainBin(explicit) {
  if (explicit && fs.existsSync(explicit)) {
    return explicit;
  }
  return candidateToolchainBins().find((candidate) => fs.existsSync(candidate)) ?? null;
}

function run(command, args, env) {
  const printable = [command, ...args].join(" ");
  console.log(printable);
  const result = spawnSync(command, args, {
    cwd: process.cwd(),
    env,
    stdio: "inherit",
  });
  if (result.status !== 0) {
    throw new Error(`Command failed with exit code ${result.status}: ${printable}`);
  }
}

function capture(command, args, env) {
  const result = spawnSync(command, args, {
    cwd: process.cwd(),
    env,
    encoding: "utf8",
    windowsHide: true,
  });
  if (result.status !== 0) return null;
  return String(result.stdout ?? "").trim();
}

function ensurePinnedGgmlSource({ buildDir, git, env }) {
  if (!git) {
    throw new Error("Unable to locate git, required to fetch the pinned ggml CPU backend.");
  }
  const dependencyRoot = path.resolve(`${buildDir}-deps`);
  const checkout = path.join(dependencyRoot, `llama.cpp-${LIZARD_GGML_COMMIT}`);
  fs.mkdirSync(checkout, { recursive: true });
  if (!fs.existsSync(path.join(checkout, ".git"))) {
    run(git, ["-C", checkout, "init"], env);
    run(git, ["-C", checkout, "remote", "add", "origin", LIZARD_GGML_REPOSITORY], env);
  }
  let commit = capture(git, ["-C", checkout, "rev-parse", "HEAD"], env);
  if (commit !== LIZARD_GGML_COMMIT) {
    run(git, ["-C", checkout, "fetch", "--depth", "1", "origin", LIZARD_GGML_COMMIT], env);
    run(git, ["-C", checkout, "checkout", "--detach", "FETCH_HEAD"], env);
    commit = capture(git, ["-C", checkout, "rev-parse", "HEAD"], env);
  }
  if (commit !== LIZARD_GGML_COMMIT) {
    throw new Error(`Pinned ggml checkout verification failed: expected ${LIZARD_GGML_COMMIT}, got ${commit ?? "unavailable"}.`);
  }
  return checkout;
}

export function nativeCpuCmakeOption(optimizeForHost = false) {
  return `-DLIZARD_NATIVE_OPTIMIZE_FOR_HOST=${optimizeForHost ? "ON" : "OFF"}`;
}

export function nativeCpuVariantCmakeOption(variant = "baseline") {
  return `-DLIZARD_NATIVE_CPU_VARIANT=${variant}`;
}

export function main() {
  const options = parseArgs(process.argv.slice(2));
  const toolchainBin = detectToolchainBin(options.toolchainBin);
  const cmake = resolveExecutable("cmake", toolchainBin);
  const gxx = resolveExecutable("g++", toolchainBin);
  const ninja = resolveExecutable("ninja", toolchainBin);
  const mingwMake = resolveExecutable("mingw32-make", toolchainBin);
  const git = resolveExecutable("git", null);

  if (!cmake) {
    throw new Error("Unable to locate cmake. Install WinLibs or set LIZARD_NATIVE_TOOLCHAIN_BIN.");
  }
  if (!gxx) {
    throw new Error("Unable to locate g++. Install WinLibs or set LIZARD_NATIVE_TOOLCHAIN_BIN.");
  }

  const generator = options.generator ?? (ninja ? "Ninja" : (mingwMake ? "MinGW Makefiles" : null));
  if (!generator) {
    throw new Error("Unable to locate Ninja or mingw32-make for the native runner build.");
  }

  const makeProgram = generator === "Ninja" ? ninja : mingwMake;
  if (!makeProgram) {
    throw new Error(`The selected generator '${generator}' requires an available build program.`);
  }

  if (options.clean) {
    for (const variant of options.variants) {
      const variantBuildDir = variant === "baseline" || variant === "host"
        ? options.buildDir
        : `${options.buildDir}-${variant}`;
      if (fs.existsSync(variantBuildDir)) {
        fs.rmSync(variantBuildDir, { recursive: true, force: true });
      }
    }
  }

  const env = {
    ...process.env,
    PATH: toolchainBin ? `${toolchainBin}${path.delimiter}${process.env.PATH ?? ""}` : (process.env.PATH ?? ""),
  };
  const ggmlSourceDir = process.env.LIZARD_GGML_SOURCE_DIR
    ? path.resolve(process.env.LIZARD_GGML_SOURCE_DIR)
    : ensurePinnedGgmlSource({ buildDir: options.buildDir, git, env });
  const binaryName = process.platform === "win32" ? "lizard-native-runner.exe" : "lizard-native-runner";
  for (const variant of options.variants) {
    const variantBuildDir = variant === "baseline" || variant === "host"
      ? options.buildDir
      : `${options.buildDir}-${variant}`;
    fs.mkdirSync(variantBuildDir, { recursive: true });
    const configureArgs = [
      "-S", options.sourceDir,
      "-B", variantBuildDir,
      "-G", generator,
      "-DCMAKE_BUILD_TYPE=Release",
      nativeCpuCmakeOption(options.optimizeForHost),
      nativeCpuVariantCmakeOption(variant === "host" ? "baseline" : variant),
      `-DCMAKE_CXX_COMPILER=${gxx}`,
      `-DCMAKE_MAKE_PROGRAM=${makeProgram}`,
    ];
    configureArgs.push(`-DLIZARD_GGML_SOURCE_DIR=${ggmlSourceDir}`);
    console.log(variant === "host"
      ? "Native CPU profile: host-optimized (development-only binary)"
      : `Native CPU profile: ${variant} (release-safe binary)`);
    run(cmake, configureArgs, env);
    run(cmake, ["--build", variantBuildDir, "--config", "Release"], env);

    const builtPath = path.join(variantBuildDir, binaryName);
    if (!fs.existsSync(builtPath)) {
      throw new Error(`Build completed but ${builtPath} was not produced.`);
    }
    const outputName = ["avx2", "avx512"].includes(variant)
      ? (process.platform === "win32" ? `lizard-native-runner-${variant}.exe` : `lizard-native-runner-${variant}`)
      : binaryName;
    const outputPath = path.join(options.buildDir, outputName);
    if (path.resolve(builtPath) !== path.resolve(outputPath)) {
      fs.mkdirSync(options.buildDir, { recursive: true });
      fs.copyFileSync(builtPath, outputPath);
    }
    console.log(`Built ${outputPath}`);
  }
}

const invokedAsScript = process.argv[1]
  && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));
if (invokedAsScript) {
  main();
}
