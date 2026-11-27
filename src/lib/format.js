import { extractEvidenceMetrics, formatEvidenceMetrics } from "./evidence-metrics.js";

export function formatHardwareSummary(hardware) {
  const gpuSummary = hardware.primaryGpu
    ? `${hardware.primaryGpu.name}${hardware.primaryGpu.vramGB ? ` (${hardware.primaryGpu.vramGB} GB VRAM)` : ""}`
    : "CPU only";
  const storage = hardware.storage
    ? `${hardware.storage.drive} free ${hardware.storage.freeGB} GB`
    : "unknown";
  return [
    "Device profile",
    `Platform: ${hardware.platform}/${hardware.arch}`,
    `CPU: ${hardware.cpu.model}`,
    `Threads: ${hardware.cpu.logicalThreads}`,
    `RAM: ${hardware.memory.totalGB} GB total, ${hardware.memory.freeGB} GB currently free, ${hardware.memory.plannerBudgetGB} GB planner budget`,
    `GPU: ${gpuSummary}`,
    `Backends: ${Object.entries(hardware.backends).filter(([, value]) => value).map(([key]) => key).join(", ")}`,
    `Storage: ${storage}`,
  ].join("\n");
}

export function formatRecommendation(recommendation) {
  const selected = recommendation.selected;
  const lines = [
    "Recommended configuration",
    `Model: ${selected.modelId} (${selected.paramsB}B)`,
    `Mode: ${selected.mode}`,
    `Quantization: ${selected.quant}`,
    `Context: ${selected.contextTokens} tokens`,
    `KV cache: ${selected.kvCache}`,
    `GPU layers: ${selected.gpuLayers}/${selected.totalLayers}`,
    `Threads: ${selected.threads}`,
    `Estimated RAM: ${selected.estimate.ramRequiredGB} GB / ${selected.estimate.allowedRamGB} GB`,
    `Estimated VRAM: ${selected.estimate.vramRequiredGB} GB / ${selected.estimate.allowedVramGB} GB`,
    "Why:",
    ...recommendation.reasons.map((reason) => `- ${reason}`),
  ];
  if (recommendation.alternatives.length > 0) {
    lines.push("Alternatives:");
    for (const alternative of recommendation.alternatives) {
      lines.push(`- ${alternative.modelId} ${alternative.quant} ${alternative.contextTokens} ctx`);
    }
  }
  return lines.join("\n");
}

export function formatExplanation(explanation) {
  return [
    "Feasibility analysis",
    `Model: ${explanation.modelId}`,
    `Verdict: ${explanation.verdict}`,
    `Quantization: ${explanation.quant}`,
    `Context: ${explanation.contextTokens}`,
    `GPU layers: ${explanation.gpuLayers}`,
    `Estimated weights: ${explanation.estimate.weightGB} GB`,
    `Estimated KV cache: ${explanation.estimate.kvCacheGB} GB`,
    `Estimated RAM: ${explanation.estimate.ramRequiredGB} GB / ${explanation.estimate.allowedRamGB} GB`,
    `Estimated VRAM: ${explanation.estimate.vramRequiredGB} GB / ${explanation.estimate.allowedVramGB} GB`,
    `Rationale: ${explanation.rationale}`,
  ].join("\n");
}

export function formatModelsList(models) {
  const lines = ["Available models"];
  for (const model of models) {
    if (model.type === "registry") {
      lines.push(`- [registry] ${model.id} (${model.paramsB}B) :: ${model.quantizations.join(", ")} :: tasks=${model.tasks.join(",")} :: ollama=${model.ollamaTag ?? "n/a"}`);
      continue;
    }
    if (model.type === "ollama") {
      lines.push(`- [ollama] ${model.tag} :: size=${model.size ?? "unknown"} :: modified=${model.modified ?? "unknown"}`);
      continue;
    }
    if (model.type === "local") {
      lines.push(`- [local] ${model.id} :: source=${model.source} :: path=${model.importedPath ?? model.path}`);
      continue;
    }
    lines.push(`- ${JSON.stringify(model)}`);
  }
  return lines.join("\n");
}

export function formatHfModelsList(payload) {
  const models = Array.isArray(payload) ? payload : payload.models;
  const lines = ["Compatible Hugging Face GGUF models (curated allowlist)"];
  if (payload.hardware) {
    lines.push(`Hardware: ${payload.hardware.gpu}, ${payload.hardware.ramGB ?? payload.hardware.plannerBudgetGB} GB RAM`);
  }
  for (const model of models) {
    lines.push(
      `- ${model.label} [${model.fit}] score=${model.compatibilityScore}` +
      `${model.taskMatch ? " (task match)" : ""}`,
    );
    lines.push(`    repo=${model.hfRepo}`);
    lines.push(`    quant=${model.recommendedQuant} file=${model.recommendedFile} ~${model.estimate.totalGB} GB RAM license=${model.license}`);
    lines.push(`    download: lizard pull ${model.hfRepo} ${model.recommendedFile}`);
  }
  return lines.join("\n");
}

export function formatHfDownloadResult(result) {
  const lines = [
    "Hugging Face model ready",
    `Model id: ${result.record.id}`,
    `Repo: ${result.record.hfRepo}`,
    `File: ${result.record.filename}`,
    `Path: ${result.record.importedPath}`,
    `sha256: ${result.record.sha256}`,
    `Verification: ${result.verification.reason}`,
    `Status: ${result.record.status}`,
  ];
  if (result.benchmark) {
    lines.push(`Benchmark: ${result.benchmark.status}${result.benchmark.tokensPerSecond != null ? ` (${result.benchmark.tokensPerSecond} tok/s)` : ""}`);
  }
  if (result.activated) {
    lines.push(`Activated: ${result.activated.id}`);
  }
  lines.push(`Next: lizard models activate --model ${result.record.id}`);
  return lines.join("\n");
}

export function formatActivateResult(active) {
  return [
    "Active model updated",
    `Model: ${active.id}`,
    `Source: ${active.source}`,
    `Path: ${active.modelPath}`,
    `Activated at: ${active.activatedAt}`,
  ].join("\n");
}

export function formatOnboardResult(result) {
  const lines = [
    "Lizard LLM — Corporate onboarding",
    "",
    "1) Checking your machine",
    `   CPU: ${result.hardware.cpu.model}`,
    `   RAM: ${result.hardware.memory.totalGB} GB (${result.hardware.memory.plannerBudgetGB} GB planner budget)`,
    `   GPU: ${result.hardware.primaryGpu?.name ?? "CPU only"}${result.hardware.primaryGpu?.vramGB ? ` (${result.hardware.primaryGpu.vramGB} GB VRAM)` : ""}`,
    "",
    "2) Finding compatible corporate models",
  ];
  result.models.slice(0, 5).forEach((model, index) => {
    lines.push(`   ${index + 1}. ${model.label} — ${model.recommendedQuant} [${model.fit}] ~${model.estimate.totalGB} GB`);
    lines.push(`      ${model.hfRepo} :: ${model.recommendedFile}`);
  });
  lines.push("");
  if (result.completed) {
    lines.push("3) Downloading selected GGUF — done");
    lines.push(`   ${result.completed.record.importedPath}`);
    lines.push("4) Verifying model integrity — " + result.completed.verification.reason);
    if (result.completed.benchmark) {
      lines.push(`5) Running benchmark — ${result.completed.benchmark.status}${result.completed.benchmark.tokensPerSecond != null ? ` (${result.completed.benchmark.tokensPerSecond} tok/s)` : ""}`);
    }
    if (result.completed.activated) {
      lines.push(`6) Model ready for Lizard — active model is ${result.completed.activated.id}`);
    }
  } else {
    lines.push("Next steps:");
    lines.push(`   Download:  lizard pull ${result.models[0]?.hfRepo} ${result.models[0]?.recommendedFile}`);
    lines.push("   Or open the guided setup:  lizard app");
  }
  return lines.join("\n");
}

export function formatPlanCommand(plan) {
  return [
    "Engine launch plan",
    `Model path: ${plan.modelPath}`,
    `Command: ${plan.displayCommand}`,
    `Executable found: ${plan.exists ? "yes" : "no"}`,
  ].join("\n");
}

export function formatGenerationResult(result) {
  const lines = [
    "Generation result",
    `Status: ${result.status}`,
    `Model: ${result.modelId}`,
    `Engine: ${result.engine ?? "llama.cpp"}`,
    `Runtime: ${result.runtime ?? "llama-cli"}`,
    `Runtime provider: ${result.runtimeProvider ?? "llama.cpp"}`,
    `Model path: ${result.modelPath}`,
    `Duration: ${result.durationMs} ms`,
  ];
  if (result.nativeBackend) {
    lines.push(`Native backend: ${result.nativeBackend}`);
  }
  if (result.nativeModelStatus) {
    lines.push(`Native model status: ${result.nativeModelStatus}`);
  }
  if (result.loadMs != null || result.prefillMs != null || result.decodeMs != null) {
    lines.push(`Native timings: load=${result.loadMs ?? "-"}ms prefill=${result.prefillMs ?? "-"}ms decode=${result.decodeMs ?? "-"}ms`);
  }
  if (result.tokensPerSecond != null) {
    lines.push(`Native tok/s: ${result.tokensPerSecond}`);
  }
  lines.push(...formatEvidenceMetrics(extractEvidenceMetrics(result)));
  if (result.note) {
    lines.push(`Note: ${result.note}`);
  }
  lines.push("Output:");
  lines.push(result.output);
  if (result.stderr) {
    lines.push("stderr:");
    lines.push(result.stderr);
  }
  return lines.join("\n");
}

export function formatEngineInstallResult(result) {
  return [
    "Engine install result",
    `Status: ${result.status}`,
    `Provider: ${result.provider}`,
    `llama-cli: ${result.cliPath ?? "not resolved"}`,
    `llama-server: ${result.serverPath ?? "not resolved"}`,
  ].join("\n");
}

function pad(value, width) {
  return String(value).padEnd(width, " ");
}
