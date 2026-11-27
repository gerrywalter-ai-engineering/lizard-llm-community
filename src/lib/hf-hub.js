// Live Hugging Face Hub client for the onboarding model selector.
//
// Unlike hf-model-catalog.js (a curated, hand-verified allowlist), this module
// talks to the real Hugging Face Hub API so the onboarding "Model Selection"
// step can browse, search, filter and paginate the entire Hub of GGUF models -
// then derive an on-device compatibility estimate WITHOUT downloading anything.
//
// Only the standard global `fetch` is used (no new dependencies). Network calls
// accept an injectable `fetchImpl` so the server endpoints and tests can stub
// the Hub. Pure parsing helpers (quant/param/pipeline detection, Link-header
// cursor extraction) are exported for direct unit testing.

import { describeModel, matchesModelFilters, normalizeTypeFilter } from "./model-taxonomy.js";

const HF_API_BASE = "https://huggingface.co/api";
const USER_AGENT = "lizard-llm/1.0 onboarding-hub";

// Bits-per-weight for footprint estimation. Superset of profiles.js's
// quantizationProfiles (which only covers the 6 quants the curated allowlist
// ships); arbitrary Hub GGUFs use many more. Values are the on-disk bpw the
// llama.cpp quant types land at, used purely to estimate RAM/disk footprint.
const QUANT_BITS = {
  TQ1_0: 1.69,
  Q2_K: 2.6,
  Q3_K_S: 3.0,
  Q3_K_M: 3.4,
  Q3_K_L: 3.7,
  Q4_0: 4.5,
  Q4_1: 4.8,
  Q4_K_S: 4.2,
  Q4_K_M: 4.5,
  Q5_0: 5.5,
  Q5_1: 6.0,
  Q5_K_S: 5.2,
  Q5_K_M: 5.5,
  Q6_K: 6.4,
  Q8_0: 8.0,
  IQ1_S: 1.6,
  IQ1_M: 1.75,
  IQ2_XXS: 2.1,
  IQ2_XS: 2.3,
  IQ2_S: 2.5,
  IQ2_M: 2.7,
  IQ3_XXS: 3.1,
  IQ3_XS: 3.3,
  IQ3_S: 3.4,
  IQ3_M: 3.6,
  IQ4_XS: 4.25,
  IQ4_NL: 4.5,
  F16: 16,
  BF16: 16,
  F32: 32,
};

// Ordering for the quant chooser: lowest-footprint first, so a user scanning
// the list sees the smallest download at the top and near-full quality last.
const QUANT_ORDER = Object.keys(QUANT_BITS).sort((a, b) => QUANT_BITS[a] - QUANT_BITS[b]);

export function quantBitsPerWeight(quant) {
  if (!quant) {
    return null;
  }
  return QUANT_BITS[String(quant).toUpperCase()] ?? null;
}

// Detect the llama.cpp quantization label from a GGUF filename, e.g.
// "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" -> "Q4_K_M". Case-insensitive;
// normalizes to the canonical uppercase label. Returns null when no known
// quant token is present.
export function detectQuantFromFilename(filename) {
  if (!filename) {
    return null;
  }
  const base = String(filename).replace(/\.gguf$/i, "");
  const upper = base.toUpperCase();
  // Prefer the most specific match (longest token) so "Q4_K_M" wins over "Q4".
  const candidates = Object.keys(QUANT_BITS)
    .slice()
    .sort((a, b) => b.length - a.length);
  for (const quant of candidates) {
    // Word-ish boundary: token must be delimited by non-alphanumerics or ends.
    const pattern = new RegExp(`(^|[^A-Z0-9])${quant.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}($|[^A-Z0-9])`);
    if (pattern.test(upper)) {
      return quant;
    }
  }
  // Bare F16/F32/BF16 sometimes written as "f16"/"fp16".
  if (/(^|[^a-z0-9])fp?16($|[^a-z0-9])/i.test(base)) {
    return "F16";
  }
  return null;
}

// Estimate parameter count in billions from a repo id / model name / tags,
// e.g. "Qwen2.5-7B-Instruct" -> 7, "Llama-3.2-1B" -> 1, "SmolLM-360M" -> 0.36,
// "Mixtral-8x7B" -> 56 (rough total for footprint). Returns null when nothing
// parseable is present. Never downloads - purely lexical.
export function estimateParamsBFromText(...texts) {
  const haystack = texts.filter(Boolean).join(" ");
  if (!haystack) {
    return null;
  }
  // MoE "NxM B" (e.g. 8x7B). Use N*M as a coarse total-params proxy.
  const moe = haystack.match(/(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*b\b/i);
  if (moe) {
    return Number((Number(moe[1]) * Number(moe[2])).toFixed(2));
  }
  // "<num>B" billions - the dominant convention.
  const billions = haystack.match(/(\d+(?:\.\d+)?)\s*b\b/i);
  if (billions) {
    const value = Number(billions[1]);
    if (Number.isFinite(value) && value > 0 && value < 2000) {
      return value;
    }
  }
  // "<num>M" millions -> billions.
  const millions = haystack.match(/(\d+(?:\.\d+)?)\s*m\b/i);
  if (millions) {
    const value = Number(millions[1]) / 1000;
    if (Number.isFinite(value) && value > 0) {
      return Number(value.toFixed(3));
    }
  }
  return null;
}

// Map a Hugging Face pipeline_tag to the onboarding filter categories.
export function pipelineCategory(pipelineTag) {
  const tag = String(pipelineTag ?? "").toLowerCase();
  if (!tag) {
    return "other";
  }
  if (tag === "text-generation" || tag === "text2text-generation") {
    return "text";
  }
  if (tag === "feature-extraction" || tag === "sentence-similarity") {
    return "embeddings";
  }
  if (tag === "image-text-to-text" || tag === "visual-question-answering" || tag === "image-to-text") {
    return "multimodal";
  }
  if (tag === "image-classification" || tag === "object-detection" || tag === "image-segmentation" || tag.startsWith("image-")) {
    return "vision";
  }
  if (tag === "automatic-speech-recognition" || tag === "text-to-speech" || tag === "audio-classification" || tag.startsWith("audio-")) {
    return "audio";
  }
  return "other";
}

// Extract a `cursor` pagination token from a Fetch `Link` header, e.g.
// `<https://huggingface.co/api/models?...&cursor=ABC>; rel="next"` -> "ABC".
// Returns null when there is no `rel="next"` link.
export function parseNextCursor(linkHeader) {
  if (!linkHeader) {
    return null;
  }
  for (const part of String(linkHeader).split(",")) {
    const match = part.match(/<([^>]+)>\s*;\s*rel="?next"?/i);
    if (match) {
      try {
        const url = new URL(match[1]);
        return url.searchParams.get("cursor");
      } catch {
        return null;
      }
    }
  }
  return null;
}

function licenseFromTags(tags = []) {
  const tag = tags.find((entry) => String(entry).toLowerCase().startsWith("license:"));
  return tag ? tag.slice("license:".length) : null;
}

function authHeaders(token) {
  const resolved = token ?? process.env.HF_TOKEN ?? process.env.HUGGING_FACE_HUB_TOKEN;
  return resolved ? { Authorization: `Bearer ${resolved}` } : {};
}

function normalizeSummary(entry) {
  const repo = entry.id ?? entry.modelId ?? "";
  const [author, ...rest] = repo.split("/");
  const name = rest.join("/") || author;
  const tags = Array.isArray(entry.tags) ? entry.tags : [];
  return {
    repo,
    id: name,
    author: repo.includes("/") ? author : (entry.author ?? null),
    name,
    downloads: entry.downloads ?? entry.downloadsAllTime ?? 0,
    likes: entry.likes ?? 0,
    pipelineTag: entry.pipeline_tag ?? null,
    category: pipelineCategory(entry.pipeline_tag),
    tags,
    license: entry.license ?? licenseFromTags(tags),
    lastModified: entry.lastModified ?? entry.last_modified ?? null,
    gated: entry.gated ?? false,
    paramsB: estimateParamsBFromText(repo, entry.pipeline_tag, tags.join(" ")),
    gguf: tags.includes("gguf") || tags.includes("GGUF"),
    // What the model IS (base vs instruct vs reasoning vs non-text), which
    // formats it ships, and which engine could run it. Discovery filters on
    // this instead of the single "is it GGUF" question it used to ask - the
    // base-vs-instruct distinction is what decides whether local chat works.
    adapter: describeModel({
      repoId: repo,
      tags,
      pipelineTag: entry.pipeline_tag ?? null,
      files: Array.isArray(entry.siblings) ? entry.siblings : [],
    }),
  };
}

const SORT_FIELDS = {
  downloads: "downloads",
  likes: "likes",
  recent: "lastModified",
  lastModified: "lastModified",
  trending: "trendingScore",
};

// Search the live Hub. Cursor-paginated: pass the `nextCursor` from a prior
// response back in to page forward. `gguf` (default true) restricts to GGUF
// repos since those are the only ones this onboarding flow can run.
export async function searchHubModels({
  query,
  author,
  pipeline,
  sort = "downloads",
  gguf = true,
  // Model-type and format filters (src/lib/model-taxonomy.js). The Hub API has
  // no "instruct vs base" facet, so these are applied to the classified results
  // after they come back rather than pushed into the query.
  types = [],
  formats = [],
  chatOnly = false,
  limit = 24,
  cursor,
  token,
  fetchImpl = fetch,
} = {}) {
  const params = new URLSearchParams();
  const search = [query, author ? `author:${author}` : null].filter(Boolean).join(" ").trim();
  if (search) {
    params.set("search", search);
  }
  if (gguf) {
    params.append("filter", "gguf");
  }
  if (pipeline) {
    params.set("pipeline_tag", pipeline);
  }
  params.set("sort", SORT_FIELDS[sort] ?? "downloads");
  params.set("direction", "-1");
  params.set("limit", String(Math.min(Math.max(Number(limit) || 24, 1), 100)));
  if (cursor) {
    params.set("cursor", cursor);
  }

  const url = `${HF_API_BASE}/models?${params.toString()}`;
  const response = await fetchImpl(url, {
    headers: { Accept: "application/json", "User-Agent": USER_AGENT, ...authHeaders(token) },
    signal: AbortSignal.timeout(12000),
  });
  if (!response.ok) {
    throw new Error(`Hugging Face search failed: HTTP ${response.status}`);
  }
  const payload = await response.json();
  const all = (Array.isArray(payload) ? payload : []).map(normalizeSummary);
  const wantedTypes = normalizeTypeFilter(types);
  const wantedFormats = (Array.isArray(formats) ? formats : [])
    .map((format) => String(format).trim().toLowerCase())
    .filter(Boolean);
  const models = (wantedTypes.length > 0 || wantedFormats.length > 0 || chatOnly)
    ? all.filter((model) => matchesModelFilters(model.adapter, {
      types: wantedTypes,
      formats: wantedFormats,
      chatOnly: Boolean(chatOnly),
    }))
    : all;
  const linkHeader = typeof response.headers?.get === "function" ? response.headers.get("link") : null;
  // `filtered` lets the caller tell "this page had none of what you asked for"
  // apart from "the Hub is out of results", which decide different UI copy.
  return {
    models,
    nextCursor: parseNextCursor(linkHeader),
    scanned: all.length,
    filtered: all.length - models.length,
  };
}

// Only single-file, runnable main-weight GGUFs belong in the quant chooser.
// Excludes:
//  - auxiliary files (mmproj / projector / vision adapters) that are NOT the
//    model weights - offering them would mislead (a 0.5 GB projector looks like
//    a tiny "Recommended" model but cannot run on its own);
//  - split/sharded GGUFs (`*-00001-of-00003.gguf`) - the single-file download
//    flow cannot assemble a multi-part model, so a lone shard is not runnable.
export function isDownloadableGguf(filename) {
  if (!filename || !/\.gguf$/i.test(filename)) {
    return false;
  }
  if (/mmproj|projector/i.test(filename)) {
    return false;
  }
  if (/-\d{2,5}-of-\d{2,5}\.gguf$/i.test(filename)) {
    return false;
  }
  return true;
}

function normalizeGgufFile(sibling) {
  const filename = sibling.rfilename ?? sibling.path ?? sibling.filename;
  const sizeBytes = sibling.size ?? sibling.lfs?.size ?? null;
  return {
    filename,
    quant: detectQuantFromFilename(filename),
    sizeBytes,
    sizeGB: sizeBytes != null ? Number((sizeBytes / 1024 ** 3).toFixed(2)) : null,
  };
}

// Full detail for one repo: model metadata + the concrete GGUF files it ships
// (with sizes and detected quant), so the UI can show a quant chooser and
// verify download readiness. Two calls: the model record (for metadata) and
// the file tree (for reliable sizes).
export async function getHubModelDetail(repo, { token, fetchImpl = fetch } = {}) {
  if (!repo) {
    throw new Error("A repository id is required.");
  }
  const headers = { Accept: "application/json", "User-Agent": USER_AGENT, ...authHeaders(token) };

  const modelResponse = await fetchImpl(`${HF_API_BASE}/models/${repo}`, {
    headers,
    signal: AbortSignal.timeout(12000),
  });
  if (!modelResponse.ok) {
    // HF returns 401 (not 404) for a missing OR private repo when
    // unauthenticated, to avoid leaking which private repos exist.
    throw new Error(
      modelResponse.status === 404 || modelResponse.status === 401
        ? `Repository not found or requires a Hugging Face token: ${repo}`
        : `Hugging Face detail failed: HTTP ${modelResponse.status}`,
    );
  }
  const model = await modelResponse.json();
  const summary = normalizeSummary(model);

  // Gather GGUF files with sizes from the recursive tree (siblings from the
  // model record often lack sizes). Fall back to siblings when the tree call
  // is unavailable.
  let ggufFiles = [];
  try {
    const treeResponse = await fetchImpl(`${HF_API_BASE}/models/${repo}/tree/main?recursive=true`, {
      headers,
      signal: AbortSignal.timeout(12000),
    });
    if (treeResponse.ok) {
      const tree = await treeResponse.json();
      ggufFiles = (Array.isArray(tree) ? tree : [])
        .filter((entry) => entry.type === "file" && isDownloadableGguf(entry.path ?? ""))
        .map(normalizeGgufFile);
    }
  } catch {
    /* fall through to siblings */
  }
  if (ggufFiles.length === 0 && Array.isArray(model.siblings)) {
    ggufFiles = model.siblings
      .filter((entry) => isDownloadableGguf(entry.rfilename ?? ""))
      .map(normalizeGgufFile);
  }

  // Deduplicate and order low-footprint first for the quant chooser.
  const seen = new Set();
  ggufFiles = ggufFiles
    .filter((file) => file.filename && !seen.has(file.filename) && seen.add(file.filename))
    .sort((a, b) => {
      const ai = a.quant ? QUANT_ORDER.indexOf(a.quant) : QUANT_ORDER.length;
      const bi = b.quant ? QUANT_ORDER.indexOf(b.quant) : QUANT_ORDER.length;
      if (ai !== bi) return ai - bi;
      return (a.sizeBytes ?? 0) - (b.sizeBytes ?? 0);
    });

  const cardData = model.cardData ?? {};
  return {
    ...summary,
    license: summary.license ?? cardData.license ?? null,
    paramsB: summary.paramsB ?? estimateParamsBFromText(cardData.model_name ?? "", (cardData.tags ?? []).join(" ")) ?? paramsFromSafetensors(model),
    contextLength: cardData.context_length ?? cardData.max_position_embeddings ?? null,
    description: typeof cardData.summary === "string" ? cardData.summary : null,
    baseModel: cardData.base_model ?? null,
    quantVariants: [...new Set(ggufFiles.map((file) => file.quant).filter(Boolean))],
    ggufFiles,
    siblingsCount: Array.isArray(model.siblings) ? model.siblings.length : ggufFiles.length,
  };
}

function paramsFromSafetensors(model) {
  const total = model?.safetensors?.total;
  if (Number.isFinite(total) && total > 0) {
    return Number((total / 1e9).toFixed(2));
  }
  return null;
}

export const __testables = { QUANT_BITS, QUANT_ORDER, normalizeSummary };
