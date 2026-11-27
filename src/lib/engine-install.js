import { spawn } from "node:child_process";
import { resolveCliExecutable, resolveServerExecutable } from "./engine-adapter.js";

function runProcess(command, args, logger) {
  return new Promise((resolve, reject) => {
    logger?.info("process-start", { command, args });
    const child = spawn(command, args, {
      stdio: "inherit",
      shell: false,
    });
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0) {
        logger?.info("process-success", { command, args });
        resolve();
        return;
      }
      reject(new Error(`${command} exited with code ${code}`));
    });
  });
}

export async function installEngine({ provider, config, logger }) {
  const selectedProvider = provider ?? config.engine.installProvider;
  if (selectedProvider !== "winget") {
    throw new Error(`Unsupported engine install provider: ${selectedProvider}`);
  }

  const existingCli = resolveCliExecutable(config);
  const existingServer = resolveServerExecutable(config);
  if (existingCli && existingServer) {
    logger?.info("engine-install-skip", {
      reason: "already-resolved",
      cliPath: existingCli,
      serverPath: existingServer,
    });
    return {
      status: "already_installed",
      provider: selectedProvider,
      cliPath: existingCli,
      serverPath: existingServer,
    };
  }

  await runProcess("winget", [
    "install",
    "--id",
    "ggml.llamacpp",
    "-e",
    "--source",
    "winget",
    "--accept-package-agreements",
    "--accept-source-agreements",
  ], logger);

  return {
    status: "installed",
    provider: selectedProvider,
    cliPath: resolveCliExecutable(config),
    serverPath: resolveServerExecutable(config),
  };
}
