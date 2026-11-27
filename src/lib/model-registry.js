import fs from "node:fs";
import { resolveAppPath } from "./app-root.js";

const registryPath = resolveAppPath("src", "data", "models.json");

export function loadModelRegistry() {
  const content = fs.readFileSync(registryPath, "utf8");
  return JSON.parse(content);
}

function modelAliases(model) {
  return Array.isArray(model.aliases) ? model.aliases : [];
}

function modelOllamaTags(model) {
  return [
    ...(model.ollamaTag ? [model.ollamaTag] : []),
    ...((Array.isArray(model.ollamaTags) ? model.ollamaTags : [])),
  ];
}

export function findModel(registry, modelId) {
  return registry.models.find((model) => model.id === modelId || modelAliases(model).includes(modelId));
}

export function findModelByOllamaTag(registry, tag) {
  return registry.models.find((model) => modelOllamaTags(model).includes(tag));
}

export function listModels(registry) {
  return registry.models.map((model) => ({
    id: model.id,
    family: model.family,
    paramsB: model.paramsB,
    tasks: model.tasks,
    quantizations: model.quantizations,
    defaultContext: model.defaultContext,
    maxContext: model.maxContext,
  }));
}

export function listRegistryModels(registry) {
  return registry.models.map((model) => ({
    id: model.id,
    aliases: modelAliases(model),
    family: model.family,
    paramsB: model.paramsB,
    tasks: model.tasks,
    quantizations: model.quantizations,
    defaultContext: model.defaultContext,
    maxContext: model.maxContext,
    ollamaTag: model.ollamaTag ?? null,
    ollamaTags: modelOllamaTags(model),
  }));
}
