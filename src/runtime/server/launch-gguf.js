import { inspectGgufFile } from "../../lib/gguf-metadata.js";

function hasTokenizerVocabulary(gguf) {
  return Array.isArray(gguf?.raw?.["tokenizer.ggml.tokens"])
    && gguf.raw["tokenizer.ggml.tokens"].length > 0;
}

export function hydrateLaunchGguf({
  modelArtifact = null,
  profile = null,
  inspect = inspectGgufFile,
} = {}) {
  const launchGguf = modelArtifact?.gguf ?? profile?.gguf ?? null;
  const needsInspection = Boolean(
    modelArtifact?.modelPath
    && !modelArtifact?.runtimePrototype
    && (launchGguf?.metadataCompacted === true || !hasTokenizerVocabulary(launchGguf)),
  );
  if (!needsInspection) {
    return launchGguf;
  }
  return inspect(modelArtifact.modelPath, {
    fallbackQuant: modelArtifact?.gguf?.quant ?? launchGguf?.quant ?? null,
    fallbackArchitecture: modelArtifact?.gguf?.architecture ?? launchGguf?.architecture ?? null,
  });
}
