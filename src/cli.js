#!/usr/bin/env node

import process from "node:process";
import { loadConfig } from "./lib/config.js";
import {
  assertRuntimeProviderAllowed,
  listRuntimeProvidersForUser,
} from "./lib/runtime-providers.js";
import { scanHardware } from "./lib/hardware.js";
import { loadModelRegistry } from "./lib/model-registry.js";
import { buildRecommendation } from "./lib/recommendation.js";
import { buildExplanation } from "./lib/explain.js";
import {
  formatActivateResult,
  formatEngineInstallResult,
  formatExplanation,
  formatGenerationResult,
  formatHardwareSummary,
  formatHfDownloadResult,
  formatHfModelsList,
  formatModelsList,
  formatOnboardResult,
  formatPlanCommand,
  formatRecommendation,
} from "./lib/format.js";
import { searchCompatibleGgufModels } from "./lib/huggingface.js";
import { runHfDownloadWorkflow } from "./lib/hf-workflow.js";
import { activateModel } from "./lib/download-manager.js";
import { buildEnginePlan, runEnginePlan } from "./lib/engine-adapter.js";
import { createCommandLogger } from "./lib/logger.js";
import { importOllamaModel, listAvailableModels } from "./lib/model-sources.js";
import { generateResponse, isEmptyCompletionOutput } from "./lib/generate.js";
import { installEngine } from "./lib/engine-install.js";
import { formatCliCommandList, formatCliHelp } from "./lib/cli-command-overview.js";
import { resolveExecutionContext } from "./lib/execution-profile.js";
import { classifyPrompt } from "./lib/prompt-classifier.js";
import { generateWithRuntimeSession, resolveRuntimeForCommand } from "./runtime/index.js";
import { buildCaterpillarServerPlan, buildNativeServerPlan, runNativeServerPlan } from "./runtime/server/run-plan.js";
import { translateOllamaStyleArgv, runOllamaStyleCommand } from "./lib/ollama-style-cli.js";
import { gpuDecodePolicyForQuant } from "./runtime/gpu-decode-policy.js";
import {
  CATERPILLAR_TTFT_PROFILE,
  applyCaterpillarPerformanceProfile,
  caterpillarPerformanceProfileFromOptions,
} from "./runtime/caterpillar-performance-profile.js";
import {
  acquireModelLoadReservation,
  buildRestartCommand,
  clearRuntimeState,
  getCurrentLoadedModel,
  getLastStoppedRuntimeModel,
  isProcessAlive,
  markCurrentModelStopped,
  markModelResidencyClaimFailed,
  monitorTrackedProcessTree,
  readModelResidencyClaim,
  refreshRuntimeState,
  registerRunningModel,
  recordTrackedProcessExit,
  releaseModelResidencyClaim,
  requireTrackedProcessTreeTermination,
  restartLastModel,
  stopCurrentModel,
} from "./lib/model-lifecycle.js";

function printHelp() {
  console.log(formatCliHelp());
}

function printCommandList() {
  console.log(formatCliCommandList());
}

// Task profile used by `recommend` and `onboard` to bias model selection.
// Derived from the explicit --task (if any) and the --prompt heuristics.
function buildTaskProfile(options) {
  return classifyPrompt({ explicitTask: options.task, prompt: options.prompt });
}

function parseArgs(argv) {
  const [command, ...rest] = argv;
  const options = {};
  for (let index = 0; index < rest.length; index += 1) {
    const token = rest[index];
    if (!token.startsWith("--")) {
      if (!options._) {
        options._ = [];
      }
      options._.push(token);
      continue;
    }

    const key = token.slice(2);
    const next = rest[index + 1];
    if (!next || next.startsWith("--")) {
      options[key] = true;
      continue;
    }

    options[key] = next;
    index += 1;
  }

  return { command, options };
}

function emit(value, asJson, formatter) {
  if (asJson) {
    console.log(JSON.stringify(value, null, 2));
    return;
  }
  console.log(formatter(value));
}

async function prepareExecution({ command, options, config, registry, logger, promptRequired = false }) {
  return resolveExecutionContext({
    command,
    options,
    config,
    registry,
    logger,
    promptRequired,
  });
}

async function withLogger(command, options, config, handler) {
  const logger = createCommandLogger({ command, options, config });
  try {
    logger.info("command-start", { command, options });
    const result = await handler(logger);
    logger.info("command-success", { command });
    return result;
  } catch (error) {
    logger.error("command-failure", {
      command,
      message: error.message,
      stack: error.stack,
    });
    throw error;
  } finally {
    logger.close();
  }
}

function snapshotProvesTrackedTreeReleased(snapshot, pid) {
  const processId = Number(pid);
  return snapshot?.capturedWhileRootAlive === true
    && Number(snapshot.rootPid) === processId
    && Array.isArray(snapshot.trackedPids)
    && snapshot.trackedPids.some((trackedPid) => Number(trackedPid) === processId)
    && snapshot.trackedPids.every((trackedPid) => !isProcessAlive(Number(trackedPid)));
}

async function finalizeCliModelProcess({
  config,
  reservation,
  pid,
  treeMonitor,
  note,
}) {
  const processId = Number(pid);
  if (!Number.isFinite(processId) || processId <= 0) {
    reservation.cancelBeforeSpawn();
    return;
  }

  // Stop sampling before termination so a registration exception cannot race
  // the root kill and erase the only pre-exit descendant inventory.
  let preExitTreeSnapshot = null;
  let monitorError = null;
  try {
    preExitTreeSnapshot = await treeMonitor?.stopAndSnapshot();
  } catch (error) {
    monitorError = error;
  }
  let treeReleaseProven = false;
  let cleanupError = null;
  if (isProcessAlive(processId)) {
    try {
      await requireTrackedProcessTreeTermination(processId);
      treeReleaseProven = true;
    } catch (error) {
      cleanupError = error;
    }
  } else {
    treeReleaseProven = snapshotProvesTrackedTreeReleased(preExitTreeSnapshot, processId);
    if (!treeReleaseProven) {
      cleanupError = Object.assign(new Error(
        `CLI model pid ${processId} exited before its complete process-tree release could be proven.`,
      ), { code: "RESIDENCY_RELEASE_FAILED", pid: processId, cause: monitorError });
    }
  }

  if (cleanupError) {
    // Preserve both ownership records. A failed cleanup must never make a
    // later CLI/dashboard process believe the model-load boundary is free.
    try {
      await recordTrackedProcessExit(config, processId, {
        note,
        preExitTreeSnapshot,
      });
    } catch {
      // recordTrackedProcessExit persists release_failed when it owns this PID.
    }
    try {
      markModelResidencyClaimFailed(config, {
        token: reservation.token,
        note: cleanupError.message,
      });
    } catch {
      // A missing/foreign claim is itself unsafe; preserve the cleanup error.
    }
    throw cleanupError;
  }

  const claim = readModelResidencyClaim(config);
  const current = getCurrentLoadedModel(config);
  if (!claim) {
    if (!current) return;
    const error = Object.assign(new Error("CLI model residency claim disappeared before lifecycle cleanup."), {
      code: "RESIDENCY_RELEASE_FAILED",
      pid: processId,
    });
    throw error;
  }
  if (claim.token !== reservation.token) {
    throw Object.assign(new Error("CLI model residency claim ownership changed during cleanup."), {
      code: "RESIDENCY_RELEASE_FAILED",
      pid: processId,
    });
  }
  if (claim.modelPid != null && Number(claim.modelPid) !== processId) {
    const error = Object.assign(new Error(
      `CLI claim pid ${claim.modelPid} does not match the verified released pid ${processId}.`,
    ), { code: "RESIDENCY_RELEASE_FAILED", pid: processId });
    markModelResidencyClaimFailed(config, { token: reservation.token, note: error.message });
    throw error;
  }
  if (current && Number(current.pid) === processId) {
    markCurrentModelStopped(config, note, processId);
  } else if (current) {
    const error = Object.assign(new Error(
      `CLI cleanup pid ${processId} does not own the tracked runtime pid ${current.pid}.`,
    ), { code: "RESIDENCY_RELEASE_FAILED", pid: processId });
    try {
      markModelResidencyClaimFailed(config, { token: reservation.token, note: error.message });
    } catch {}
    throw error;
  }

  if (claim.modelPid == null) {
    if (!reservation.cancelBeforeSpawn()) {
      throw Object.assign(new Error("CLI could not cancel its PID-less model reservation after verified cleanup."), {
        code: "RESIDENCY_RELEASE_FAILED",
        pid: processId,
      });
    }
    return;
  }
  releaseModelResidencyClaim(config, {
    token: reservation.token,
    pid: processId,
    treeReleaseProven: true,
  });
}

async function main() {
  // Ollama-style verbs (pull, run, list, ps, stop, rm, show, serve, app, pro)
  // are translated onto the commands below, or handled directly when they
  // have no older equivalent (the interactive `run`, `rm`, `show`, `pro`).
  const rawArgv = translateOllamaStyleArgv(process.argv.slice(2));
  if (await runOllamaStyleCommand(rawArgv)) {
    return;
  }
  const { command, options } = parseArgs(rawArgv);

  if (!command || command === "help" || command === "-help" || command === "--help" || options.help) {
    printHelp();
    return;
  }

  if (command === "commands" || command === "-commands" || command === "--commands") {
    printCommandList();
    return;
  }

  const config = loadConfig();
  const registry = loadModelRegistry();

  if (command === "scan") {
    await withLogger(command, options, config, async (logger) => {
      const hardware = scanHardware({ config, logger });
      emit(hardware, options.json, formatHardwareSummary);
    });
    return;
  }

  if (command === "recommend") {
    await withLogger(command, options, config, async (logger) => {
      if (options["list-models"]) {
        const models = await listAvailableModels({
          config,
          registry,
          source: options.source ?? "all",
          logger,
        });
        emit(models, options.json, formatModelsList);
        return;
      }

      const hardware = scanHardware({ config, logger });
      const taskProfile = buildTaskProfile(options);
      const recommendation = buildRecommendation({
        hardware,
        registry,
        config,
        requestedMode: options.mode,
        requestedModelId: options.model,
        taskProfile,
      });
      emit(recommendation, options.json, formatRecommendation);
    });
    return;
  }

  if (command === "explain") {
    await withLogger(command, options, config, async (logger) => {
      if (!options.model) {
        throw new Error("--model is required for explain");
      }

      const hardware = scanHardware({ config, logger });
      const explanation = buildExplanation({
        registry,
        hardware,
        config,
        modelId: options.model,
        quant: options.quant,
        contextTokens: options.context ? Number.parseInt(options.context, 10) : undefined,
        mode: options.mode,
      });
      emit(explanation, options.json, formatExplanation);
    });
    return;
  }

  if (command === "models") {
    await withLogger(command, options, config, async (logger) => {
      const subcommand = options._?.[0];
      if (!subcommand || subcommand === "list") {
        const models = await listAvailableModels({
          config,
          registry,
          source: options.source ?? "all",
          logger,
        });
        emit(models, options.json, formatModelsList);
        return;
      }

      if (subcommand === "hf-search") {
        const hardware = scanHardware({ config, logger });
        const models = searchCompatibleGgufModels({
          hardware,
          task: options.task,
          mode: options.mode ?? config.defaultMode,
        });
        emit({ hardware: {
          gpu: hardware.primaryGpu?.name ?? "CPU only",
          ramGB: hardware.memory.totalGB,
          plannerBudgetGB: hardware.memory.plannerBudgetGB,
        }, models }, options.json, formatHfModelsList);
        return;
      }

      if (subcommand === "hf-download") {
        if (!options.repo || !options.file) {
          throw new Error("--repo and --file are required for hf-download");
        }
        const hardware = scanHardware({ config, logger });
        const workflow = await runHfDownloadWorkflow({
          repo: options.repo,
          filename: options.file,
          modelId: options.alias,
          token: options.token,
          config,
          logger,
          onPhase: (phase) => {
            if (!options.json) {
              console.error(`  ${phase.message}…`);
            }
          },
        });
        const output = { record: workflow.record, verification: workflow.verification };
        if (options.activate) {
          output.activated = activateModel({ config, modelId: workflow.record.id });
        }
        emit(output, options.json, formatHfDownloadResult);
        return;
      }

      if (subcommand === "activate") {
        if (!options.model) {
          throw new Error("--model MODEL_ID is required for activate");
        }
        const active = activateModel({ config, modelId: options.model });
        emit(active, options.json, formatActivateResult);
        return;
      }

      if (subcommand === "import" || subcommand === "pull") {
        if (!options.ollama) {
          throw new Error("--ollama TAG is required");
        }
        const result = await importOllamaModel({
          tag: options.ollama,
          alias: options.alias,
          strategy: options.strategy,
          pullIfMissing: subcommand === "pull",
          config,
          registry,
          logger,
        });
        emit(result.models, options.json, formatModelsList);
        return;
      }

      throw new Error(`Unknown models subcommand: ${subcommand}`);
    });
    return;
  }

  if (command === "model") {
    await withLogger(command, options, config, async () => {
      const subcommand = options._?.[0] ?? "status";
      if (subcommand === "status") {
        const state = refreshRuntimeState(config);
        emit({
          currentLoadedModel: state.currentLoadedModel,
          lastStoppedModel: state.lastStoppedModel,
          restartCommand: buildRestartCommand(state.lastStoppedModel),
        }, options.json, (payload) => [
          "Model lifecycle status",
          `Current: ${payload.currentLoadedModel?.modelId ?? "none"}${payload.currentLoadedModel?.pid ? ` (pid ${payload.currentLoadedModel.pid})` : ""}`,
          `Last stopped: ${payload.lastStoppedModel?.modelId ?? "none"}`,
          `Restart command: ${payload.restartCommand ?? "n/a"}`,
        ].join("\n"));
        return;
      }
      if (subcommand === "stop") {
        // Unloading kills the server process, and with it every response still
        // streaming to a chat window or an API client. Say what is about to be
        // cut off and get an answer first. --yes keeps this usable from scripts
        // and from the benchmark pre-run clear, which has already decided.
        const { describeUnloadImpact, unloadImpactMessage } = await import("./lib/model-activity.js");
        const impact = await describeUnloadImpact(config);
        const warning = unloadImpactMessage(impact);
        if (warning && !options.yes && !options.force && !options.json) {
          console.log(warning);
          const { createInterface } = await import("node:readline/promises");
          const rl = createInterface({ input: process.stdin, output: process.stdout });
          const answer = (await rl.question("Type y to unload anyway: ")).trim().toLowerCase();
          rl.close();
          if (answer !== "y" && answer !== "yes") {
            console.log("Left the model loaded. Nothing was interrupted.");
            return;
          }
        } else if (warning && options.json) {
          // A JSON caller cannot be prompted, so the impact travels in the
          // payload instead of being silently discarded.
          process.stdout.write("");
        }
        const result = await stopCurrentModel(config);
        if (warning) result.unloadImpact = impact;
        emit(result, options.json, (payload) => payload.stopped
          ? (payload.reason === "already_exited"
            ? `Tracked model process had already exited; recorded as stopped. Last stopped: ${payload.state?.lastStoppedModel?.modelId ?? "unknown"}`
            : `Stopped current model. Last stopped: ${payload.state?.lastStoppedModel?.modelId ?? "unknown"}`)
          : "No tracked running model to stop.");
        return;
      }
      if (subcommand === "restart-last") {
        const result = restartLastModel(config);
        emit(result, options.json, (payload) => `Restarted ${payload.currentLoadedModel?.modelId ?? "model"} (pid ${payload.currentLoadedModel?.pid ?? "unknown"}).`);
        return;
      }
      if (subcommand === "clear-state") {
        const result = clearRuntimeState(config);
        emit(result, options.json, () => "Cleared runtime lifecycle state.");
        return;
      }
      throw new Error(`Unknown model subcommand: ${subcommand}`);
    });
    return;
  }

  if (command === "engine") {
    await withLogger(command, options, config, async (logger) => {
      const subcommand = options._?.[0];
      if (subcommand !== "install") {
        throw new Error("Only 'engine install' is currently supported");
      }
      const result = await installEngine({
        provider: options.provider ?? config.engine.installProvider,
        config,
        logger,
      });
      emit(result, options.json, formatEngineInstallResult);
    });
    return;
  }

  if (command === "launch") {
    await withLogger(command, options, config, async (logger) => {
      const runtimeResolution = resolveRuntimeForCommand({
        command: "run",
        requestedEngine: options.engine ?? null,
      });
      if (runtimeResolution.experimental) {
        // Community Edition: both native engines are free.
      }
      const { recommendation, modelArtifact, profile } = await prepareExecution({
        command: "run",
        options,
        config,
        registry,
        logger,
      });
      const runtimeProfile = applyCaterpillarPerformanceProfile(profile, {
        runtimeProvider: runtimeResolution.engine,
        performanceProfile: caterpillarPerformanceProfileFromOptions(options),
      });
      const plan = buildEnginePlan({
        recommendation,
        profile: runtimeProfile,
        config,
        enginePath: options["engine-path"],
        modelPathOverride: modelArtifact.modelPath,
      });
      const standalonePlan = runtimeResolution.experimental
        ? (runtimeResolution.engine === "caterpillar" ? buildCaterpillarServerPlan : buildNativeServerPlan)({
          config,
          recommendation,
          modelArtifact,
          profile: runtimeProfile,
        })
        : null;

      if (options["dry-run"]) {
        console.log(formatPlanCommand(runtimeResolution.experimental
          ? standalonePlan
          : plan));
        return;
      }

      const modelLoadReservation = acquireModelLoadReservation(config, {
        key: {
          provider: runtimeResolution.experimental ? runtimeResolution.engine : "llama.cpp",
          modelPath: modelArtifact.modelPath,
          quant: runtimeProfile.gguf?.quant
            ?? modelArtifact.sourceQuantType
            ?? modelArtifact.gguf?.quant
            ?? recommendation.selected?.quant,
        },
      });

      let trackedPid = null;
      let trackedTreeMonitor = null;
      try {
        const runner = runtimeResolution.experimental
          ? runNativeServerPlan(
            standalonePlan,
            {
              onSpawn: (child) => {
                trackedPid = child.pid;
                trackedTreeMonitor = monitorTrackedProcessTree(child.pid);
                modelLoadReservation.attachModel(child.pid);
                registerRunningModel(config, {
                  modelId: modelArtifact.modelId,
                  pid: child.pid,
                  profile: runtimeProfile,
                  command: "native-server",
                  executable: process.execPath,
                  args: [],
                  source: runtimeResolution.engine === "caterpillar" ? "cli-run-caterpillar" : "cli-run-native",
                  claimToken: modelLoadReservation.token,
                });
              },
              callerOwnsCleanup: true,
            },
          )
          : runEnginePlan(plan, logger, {
          onSpawn: (child) => {
            trackedPid = child.pid;
            trackedTreeMonitor = monitorTrackedProcessTree(child.pid);
            modelLoadReservation.attachModel(child.pid);
            registerRunningModel(config, {
              modelId: modelArtifact.modelId,
              pid: child.pid,
              profile,
              command: plan.displayCommand,
              executable: plan.executable,
              args: plan.args,
              source: "cli-run",
              claimToken: modelLoadReservation.token,
            });
          },
          callerOwnsCleanup: true,
        });
        await runner;
      } finally {
        await finalizeCliModelProcess({
          config,
          reservation: modelLoadReservation,
          pid: trackedPid,
          treeMonitor: trackedTreeMonitor,
          note: "CLI run process exited and its model process tree was verified released.",
        });
      }
    });
    return;
  }

  if (command === "serve") {
    await withLogger(command, options, config, async (logger) => {
      // Community Edition: the native API server is free and needs no licence.
      const requestedProvider = String(options.provider ?? "lizard-native").trim().toLowerCase();
      // The CLI runs as the machine operator; the provider list stays the single source.
      const serveProviders = listRuntimeProvidersForUser({ isAdmin: true, local: true })
        .filter((provider) => ["lizard-native", "caterpillar"].includes(provider.id))
        .map((provider) => provider.id);
      if (!serveProviders.includes(requestedProvider)) {
        throw new Error(`--provider for serve must be one of: ${serveProviders.join(", ")}.`);
      }
      assertRuntimeProviderAllowed(requestedProvider, { isAdmin: true, local: true });
      const serveOptions = {
        ...options,
        model: options.model ?? "active",
        backend: "cpu",
        "gpu-layers": "0",
        engine: requestedProvider,
      };
      const { recommendation, modelArtifact, profile } = await prepareExecution({
        command: "run",
        options: serveOptions,
        config,
        registry,
        logger,
      });
      const cpuOnly = Boolean(options["cpu-only"]);
      const nativeProfile = {
        ...profile,
        backend: "cpu",
        gpuLayers: 0,
        timeoutMs: options.timeout == null || Number(options.timeout) <= 0
          ? null
          : profile.timeoutMs,
        nativeRuntime: {
          ...(profile.nativeRuntime ?? {}),
          allowBackendFallback: false,
          keepWorkerLoaded: true,
          // The native worker offloads prefill/decode projections to the
          // Direct3D 12 resident-weight lane when the execution plan allows
          // it; --cpu-only forces the plan (and therefore the whole request
          // path) onto the CPU.
          enableHybridGpuPrefill: !cpuOnly,
          forceCpuOnlyPlan: cpuOnly,
          gpuDecodePolicy: gpuDecodePolicyForQuant(
            profile.gguf?.quant ?? modelArtifact.gguf?.quant,
            { gpuEnabled: !cpuOnly },
          ),
        },
      };
      // Caterpillar performance profiles are a no-op for the other engines -
      // applyCaterpillarPerformanceProfile returns the profile untouched unless
      // the provider is caterpillar.
      let selectedProfile = applyCaterpillarPerformanceProfile(nativeProfile, {
        runtimeProvider: requestedProvider,
        performanceProfile: caterpillarPerformanceProfileFromOptions(options),
      });
      if (cpuOnly) {
        selectedProfile = {
          ...selectedProfile,
          nativeRuntime: {
            ...(selectedProfile.nativeRuntime ?? {}),
            enableHybridGpuPrefill: false,
            forceCpuOnlyPlan: true,
            gpuDecodePolicy: "disabled",
          },
        };
      }
      const planBuilder = requestedProvider === "caterpillar" ? buildCaterpillarServerPlan : buildNativeServerPlan;
      const plan = planBuilder({
        config,
        recommendation,
        modelArtifact,
        profile: selectedProfile,
        host: options.host ?? null,
        port: options.port ?? null,
        internalBenchmarkServer: process.env.LIZARD_INTERNAL_HTTP_BENCHMARK === "1",
      });

      if (options["dry-run"]) {
        console.log(formatPlanCommand(plan));
        return;
      }

      const modelLoadReservation = acquireModelLoadReservation(config, {
        key: {
          provider: requestedProvider,
          modelPath: modelArtifact.modelPath,
          quant: selectedProfile.gguf?.quant
            ?? modelArtifact.sourceQuantType
            ?? modelArtifact.gguf?.quant
            ?? recommendation.selected?.quant,
        },
      });

      const baseUrl = `http://${plan.host}:${plan.port}`;
      const gpuName = selectedProfile.capabilityRegistry?.gpu?.[0]?.name ?? null;
      const runtimeLabel = cpuOnly || !gpuName
        ? `${requestedProvider} / CPU / no fallback`
        : `${requestedProvider} / GPU hybrid (${gpuName}) / explicit CPU fallback`;
      const providerLabel = requestedProvider === "caterpillar" ? "Caterpillar" : "Lizard Native";
      console.log(`${providerLabel} API server (Lizard LLM Community Edition)`);
      console.log(`  Model: ${modelArtifact.modelId}`);
      console.log(`  Runtime: ${runtimeLabel}`);
      console.log(`  Request timeout: ${selectedProfile.timeoutMs == null ? "unlimited" : `${selectedProfile.timeoutMs} ms`}`);
      if (requestedProvider === "caterpillar") {
        console.log(`  Caterpillar profile: ${selectedProfile.caterpillarPerformanceProfile ?? "balanced"}`);
      }
      console.log(`  Chat completions: ${baseUrl}/v1/chat/completions`);
      console.log(`  Health: ${baseUrl}/health`);
      console.log("  Press Ctrl+C to stop.");
      logger.info(`${requestedProvider}-api-server-start`, {
        modelId: modelArtifact.modelId,
        host: plan.host,
        port: plan.port,
      });

      let trackedPid = null;
      let trackedTreeMonitor = null;
      try {
        await runNativeServerPlan(plan, {
          onSpawn: (child) => {
            trackedPid = child.pid;
            trackedTreeMonitor = monitorTrackedProcessTree(child.pid);
            modelLoadReservation.attachModel(child.pid);
            registerRunningModel(config, {
              modelId: modelArtifact.modelId,
              pid: child.pid,
              profile: selectedProfile,
              command: `${requestedProvider}-api-server`,
              executable: plan.executable,
              args: plan.args,
              source: requestedProvider === "caterpillar" ? "cli-serve-caterpillar" : "cli-serve-native",
              claimToken: modelLoadReservation.token,
            });
          },
          callerOwnsCleanup: true,
        });
      } finally {
        await finalizeCliModelProcess({
          config,
          reservation: modelLoadReservation,
          pid: trackedPid,
          treeMonitor: trackedTreeMonitor,
          note: "CLI serve process exited and its model process tree was verified released.",
        });
      }
    });
    return;
  }

  if (command === "generate") {
    await withLogger(command, options, config, async (logger) => {
      if (!options.prompt) {
        throw new Error("--prompt is required for generate");
      }

      const { recommendation, modelArtifact, profile } = await prepareExecution({
        command: "generate",
        options,
        config,
        registry,
        logger,
      });
      const maxTokens = options["max-tokens"] ? Number.parseInt(options["max-tokens"], 10) : config.generation.maxTokens;
      const runtimeResolution = resolveRuntimeForCommand({
        command: "generate",
        requestedEngine: options.engine ?? null,
      });
      const result = runtimeResolution.experimental
        ? await generateWithRuntimeSession({
          requestedEngine: options.engine ?? null,
          recommendation,
          modelArtifact,
          profile: applyCaterpillarPerformanceProfile(profile, {
            runtimeProvider: runtimeResolution.engine,
            performanceProfile: caterpillarPerformanceProfileFromOptions(options),
          }),
          prompt: options.prompt,
          systemPrompt: options.system,
          maxTokens,
          enginePath: options["engine-path"],
          config,
          logger,
        })
        : await generateResponse({
          recommendation,
          modelArtifact,
          profile,
          prompt: options.prompt,
          systemPrompt: options.system,
          maxTokens,
          enginePath: options["engine-path"],
          config,
          logger,
        });

      let finalResult = result;
      // Some GGUF conversions carry a misclassified special token that makes
      // llama.cpp's vocab loader fold it into the EOG set (in place of the
      // real EOS token), so generation stops immediately and llama.cpp's own
      // completion->cli retry (in generateResponse) still comes back empty.
      // lizard-native has its own tokenizer path and isn't affected, so retry
      // there once before giving up.
      if (!runtimeResolution.experimental && isEmptyCompletionOutput(result.output)) {
        logger.warn("generate-empty-llamacpp-output-retrying-lizard-native", {
          modelId: modelArtifact.modelId,
          runtime: result.runtime,
        });
        try {
          const nativeResult = await generateWithRuntimeSession({
            requestedEngine: "lizard-native",
            recommendation,
            modelArtifact,
            profile,
            prompt: options.prompt,
            systemPrompt: options.system,
            maxTokens,
            enginePath: options["engine-path"],
            config,
            logger,
          });
          if (!isEmptyCompletionOutput(nativeResult.output)) {
            finalResult = {
              ...nativeResult,
              note: "llama.cpp returned an empty completion for this model (a known vocab-loading quirk where a special token gets folded into the EOG set); retried automatically via the lizard-native engine.",
            };
          }
        } catch (nativeError) {
          logger.warn("generate-lizard-native-fallback-failed", { message: nativeError.message });
        }
      }
      emit(finalResult, options.json, formatGenerationResult);
    });
    return;
  }

  if (command === "onboard") {
    await withLogger(command, options, config, async (logger) => {
      const hardware = scanHardware({ config, logger });
      const taskProfile = buildTaskProfile(options);
      const mode = options.mode ?? taskProfile.recommendedMode ?? config.defaultMode;
      const models = searchCompatibleGgufModels({
        hardware,
        task: options.task,
        mode,
      });

      let completed = null;
      if (options.repo && options.file) {
        const workflow = await runHfDownloadWorkflow({
          repo: options.repo,
          filename: options.file,
          modelId: options.alias,
          token: options.token,
          config,
          logger,
          onPhase: (phase) => {
            if (!options.json) {
              console.error(`  ${phase.message}…`);
            }
          },
        });
        completed = {
          record: workflow.record,
          verification: workflow.verification,
        };
        if (options.activate) {
          completed.activated = activateModel({ config, modelId: workflow.record.id });
        }
      }

      emit({ hardware, models, completed }, options.json, formatOnboardResult);
    });
    return;
  }

  if (command === "serve-ui") {
    await withLogger(command, options, config, async (logger) => {
      const host = options.host ?? "127.0.0.1";
      const uiCaterpillarProfile = caterpillarPerformanceProfileFromOptions(options);
      if (uiCaterpillarProfile === CATERPILLAR_TTFT_PROFILE) {
        process.env.LIZARD_CATERPILLAR_PROFILE = CATERPILLAR_TTFT_PROFILE;
      }

      // --cpu-only pins every native worker this app spawns to the processor
      // engine - the Enhancement Run 06 pinned ggml CPU backend. Lizard routes
      // a GPU-eligible quant (Q4_K_M and friends) to the resident Direct3D 12
      // lane by default, so without this the UI never exercises the CPU engine
      // and chat/benchmarks report directx-hybrid. These two switches are what
      // the runner itself honours; the flag exists so nobody has to know them.
      const cpuOnly = Boolean(options["cpu-only"]);
      const cpuOnlyEnv = cpuOnly
        ? { LIZARD_NATIVE_GPU_DECODE: "0", LIZARD_NATIVE_GPU_PRESTAGE: "0" }
        : {};
      if (cpuOnly) {
        Object.assign(process.env, cpuOnlyEnv);
      }

      // One local app: onboarding at / and /onboarding, chat inside it.
      const { startUiServer } = await import("./ui/server.js");
      const port = options.port ? Number.parseInt(options.port, 10) : Number.parseInt(process.env.LIZARD_PORT ?? "7817", 10);
      const { url } = await startUiServer({ config, logger, port, host });
      console.log(`Lizard LLM Community Edition running at ${url}`);
      console.log(cpuOnly
        ? "  Engine: processor only (pinned ggml CPU backend)"
        : "  Engine: automatic (Direct3D 12 lane when the quant supports it)");
      console.log("Press Ctrl+C to stop.");
      logger.info("serve-ui-start", { url, cpuOnly });
      if (options.open) {
        const { spawn } = await import("node:child_process");
        const opener = process.platform === "win32" ? ["cmd", ["/c", "start", "", url]]
          : process.platform === "darwin" ? ["open", [url]] : ["xdg-open", [url]];
        spawn(opener[0], opener[1], { stdio: "ignore", detached: true }).unref();
      }
      await new Promise(() => {});
    });
    return;
  }

  throw new Error(`Unknown command: ${command}`);
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = error.exitCode ?? 1;
});
