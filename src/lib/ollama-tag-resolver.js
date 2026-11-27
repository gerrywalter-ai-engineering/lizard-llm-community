import { findModel, findModelByOllamaTag } from "./model-registry.js";
import { listLocalCatalogModels } from "./model-sources.js";
import { ensureOllamaPulled, listInstalledOllamaModels } from "./ollama.js";

// Resolves any model reference (curated registry id, local catalog id, or raw
// Ollama tag) to an Ollama tag that can actually be served. The curated
// registry (src/data/models.json) only lists a handful of models; models
// onboarded through the dashboard live in the runtime catalog
// (~/.lizard-llm/model-catalog.json) with their own `ollamaTag`. Passing a
// catalog id such as "qwen2.5-3b" straight to `ollama pull` fails with
// "pull model manifest: file does not exist" because the Ollama registry only
// knows "qwen2.5:3b" (see the 2026-07-10 benchmark-best-q-single failure).

function normalizeTagKey(value) {
  return String(value ?? "")
    .toLowerCase()
    .replace(/:latest$/, "")
    .replace(/[^a-z0-9]+/g, "");
}

const IDENTITY_NOISE_TOKENS = new Set([
  "instruct", "chat", "it", "base", "gguf", "hf", "gptq", "awq",
]);

// Quant designators split on separators ("Q4_K_M" -> "q4"/"k"/"m"), so the
// short letter fragments and bare digits they leave behind are noise too.
const QUANT_TOKEN_PATTERN = /^(q\d.*|iq\d.*|tq\d.*|f16|f32|bf16|fp16|fp32|[klmsx]{1,3}|\d+)$/;

// Extracts a comparable identity from a model reference: the family tokens
// ("qwen2.5") and the parameter size in billions (3). Works for catalog ids
// ("qwen2.5-3b"), Ollama tags ("qwen2.5:3b"), and HF-style names
// ("Qwen2.5-3B-Instruct-Q4_K_M").
export function modelIdentity(value) {
  const tokens = String(value ?? "")
    .toLowerCase()
    .split(/[^a-z0-9.]+/)
    .filter(Boolean);
  let sizeB = null;
  const familyTokens = [];
  for (const token of tokens) {
    const sizeMatch = token.match(/^(\d+(?:\.\d+)?)b$/);
    if (sizeMatch && sizeB == null) {
      sizeB = Number(sizeMatch[1]);
      continue;
    }
    if (IDENTITY_NOISE_TOKENS.has(token) || QUANT_TOKEN_PATTERN.test(token)) {
      continue;
    }
    familyTokens.push(token);
  }
  return {
    key: normalizeTagKey(value),
    family: familyTokens.join("-"),
    sizeB,
  };
}

function catalogEntryForRef(config, modelRef) {
  if (!config || !modelRef) {
    return null;
  }
  let models = [];
  try {
    models = listLocalCatalogModels(config) ?? [];
  } catch {
    return null;
  }
  const wanted = String(modelRef);
  return models.find((model) => model.id === wanted)
    ?? models.find((model) => Array.isArray(model.aliases) && model.aliases.includes(wanted))
    ?? models.find((model) => model.ollamaTag === wanted)
    ?? null;
}

// Ordered, de-duplicated candidate tags for a model reference. Sources:
// the reference itself when it already looks like a tag, the curated
// registry, and the runtime model catalog.
export function ollamaTagCandidates({ modelRef, registry = null, config = null }) {
  const candidates = [];
  const push = (tag, source) => {
    if (tag && !candidates.some((entry) => entry.tag === tag)) {
      candidates.push({ tag, source });
    }
  };

  if (String(modelRef ?? "").includes(":")) {
    push(String(modelRef), "direct");
  }
  if (registry) {
    push(findModel(registry, modelRef)?.ollamaTag, "registry");
    if (findModelByOllamaTag(registry, modelRef)) {
      push(String(modelRef), "registry");
    }
  }
  push(catalogEntryForRef(config, modelRef)?.ollamaTag, "catalog");
  return candidates;
}

// Finds an installed tag that matches the reference by identity: exact
// normalized equality first ("qwen2.5-3b" == "qwen2.5:3b"), then same
// family and same parameter size ("qwen2.5-3b-instruct" -> "qwen2.5:3b").
export function matchInstalledOllamaTag(modelRef, installedTags) {
  const wanted = modelIdentity(modelRef);
  if (!wanted.key) {
    return null;
  }
  const exact = installedTags.find((tag) => normalizeTagKey(tag) === wanted.key);
  if (exact) {
    return { tag: exact, match: "exact" };
  }
  if (!wanted.family || wanted.sizeB == null) {
    return null;
  }
  const sameIdentity = installedTags.find((tag) => {
    const candidate = modelIdentity(tag);
    return candidate.family === wanted.family && candidate.sizeB === wanted.sizeB;
  });
  return sameIdentity ? { tag: sameIdentity, match: "family-size" } : null;
}

function installedTagList(installedTags, logger) {
  if (Array.isArray(installedTags)) {
    return installedTags;
  }
  return listInstalledOllamaModels(logger).map((model) => model.tag).filter(Boolean);
}

// Non-pulling resolution: returns the best tag for the reference, preferring
// tags that are already installed. Returns null when nothing matches.
export function resolveOllamaTag({ modelRef, registry = null, config = null, installedTags = null, logger = null }) {
  const candidates = ollamaTagCandidates({ modelRef, registry, config });
  const installed = installedTagList(installedTags, logger);

  for (const candidate of candidates) {
    const hit = installed.find((tag) => normalizeTagKey(tag) === normalizeTagKey(candidate.tag));
    if (hit) {
      return { tag: hit, source: candidate.source, installed: true, substituted: false };
    }
  }
  const fuzzy = matchInstalledOllamaTag(modelRef, installed);
  if (fuzzy) {
    return { tag: fuzzy.tag, source: `installed-${fuzzy.match}`, installed: true, substituted: fuzzy.match !== "exact" };
  }
  if (candidates.length > 0) {
    return { tag: candidates[0].tag, source: candidates[0].source, installed: false, substituted: false };
  }
  return null;
}

// Full resolution ladder used by the benchmark before its Ollama lane:
//   1. a known candidate tag (direct / registry / catalog) that is already
//      installed wins outright - no download;
//   2. an installed model matching the requested family + parameter size is
//      reused instead of downloading a duplicate;
//   3. otherwise each candidate is pulled in order until one succeeds;
//   4. only when every route fails does this throw, listing every attempt so
//      the failure is diagnosable instead of a bare manifest error.
export function ensureOllamaModelAvailable({
  modelRef,
  registry = null,
  config = null,
  installedTags = null,
  logger = null,
  pull = null,
}) {
  const candidates = ollamaTagCandidates({ modelRef, registry, config });
  const installed = installedTagList(installedTags, logger);
  const attempts = [];

  for (const candidate of candidates) {
    const hit = installed.find((tag) => normalizeTagKey(tag) === normalizeTagKey(candidate.tag));
    if (hit) {
      logger?.info("ollama-tag-resolved", { modelRef, tag: hit, source: candidate.source, installed: true });
      return { tag: hit, source: candidate.source, installed: true, pulled: false, substituted: false };
    }
  }

  const fuzzy = matchInstalledOllamaTag(modelRef, installed);
  if (fuzzy) {
    logger?.info("ollama-tag-resolved", {
      modelRef,
      tag: fuzzy.tag,
      source: `installed-${fuzzy.match}`,
      installed: true,
      substituted: fuzzy.match !== "exact",
    });
    return {
      tag: fuzzy.tag,
      source: `installed-${fuzzy.match}`,
      installed: true,
      pulled: false,
      substituted: fuzzy.match !== "exact",
    };
  }

  const pullTag = pull ?? ((tag) => ensureOllamaPulled(tag, config, logger));
  for (const candidate of candidates) {
    try {
      logger?.info("ollama-tag-pull-attempt", { modelRef, tag: candidate.tag, source: candidate.source });
      pullTag(candidate.tag);
      logger?.info("ollama-tag-resolved", { modelRef, tag: candidate.tag, source: candidate.source, pulled: true });
      return { tag: candidate.tag, source: candidate.source, installed: false, pulled: true, substituted: false };
    } catch (error) {
      attempts.push(`pull ${candidate.tag} (${candidate.source}): ${error.message}`);
      logger?.warn("ollama-tag-pull-failed", { modelRef, tag: candidate.tag, message: error.message });
    }
  }

  const attemptSummary = attempts.length > 0 ? ` Attempts: ${attempts.join("; ")}` : "";
  throw new Error(
    `Unable to resolve an Ollama model for "${modelRef}": no registry/catalog tag is installed, `
    + `no installed model matches its family and size, and every pull candidate failed.${attemptSummary} `
    + `Install a matching model manually (e.g. "ollama pull <tag>") or pick a different benchmark model.`,
  );
}
