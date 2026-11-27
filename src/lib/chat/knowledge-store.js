// Workspace knowledge: documents you drop in, chunked and retrievable, so the
// model can answer from your own material with citations.
//
// Retrieval is BM25 over the workspace's chunks. That is a deliberate choice,
// not a shortcut: an embedding index would need a second model resident
// alongside the chat model, and Lizard's whole point is running on hardware
// that barely fits one. BM25 needs no model, no GPU and no network, indexes
// instantly, and on the "find the paragraph that mentions this" workload a
// chat sidebar actually has, it is competitive with dense retrieval.
//
// Storage: <runtimeDirectory>/chat/knowledge/<workspaceId>/<docId>.json

import fs from "node:fs";
import path from "node:path";
import { ensureDirectory } from "../paths.js";
import { newId, safeId } from "./chat-store.js";

const CHUNK_TARGET_CHARS = 900;
const CHUNK_OVERLAP_CHARS = 140;
const MAX_DOCUMENT_CHARS = 2_000_000;

// Formats we can read as text without pulling in a parser. Binary office and
// PDF files are rejected with an explicit message rather than indexed as mojibake.
export const SUPPORTED_EXTENSIONS = Object.freeze([
  ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv", ".json", ".jsonl",
  ".yaml", ".yml", ".toml", ".ini", ".env", ".xml", ".html", ".htm",
  ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".py", ".rb", ".go", ".rs",
  ".java", ".kt", ".c", ".h", ".cpp", ".hpp", ".cs", ".sh", ".ps1", ".sql",
]);

const STOP_WORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
  "how", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the",
  "their", "then", "there", "these", "they", "this", "to", "was", "were",
  "what", "when", "where", "which", "who", "why", "will", "with", "you", "your",
]);

function knowledgeRoot(config) {
  return path.join(config?.runtimeDirectory ?? process.cwd(), "chat", "knowledge");
}

function workspaceDirectory(config, workspaceId) {
  return path.join(knowledgeRoot(config), safeId(workspaceId));
}

function documentPath(config, workspaceId, documentId) {
  return path.join(workspaceDirectory(config, workspaceId), `${safeId(documentId)}.json`);
}

export function tokenize(text) {
  return String(text ?? "")
    .toLowerCase()
    .split(/[^a-z0-9_À-ɏ]+/i)
    .filter((token) => token.length > 1 && token.length < 40 && !STOP_WORDS.has(token));
}

// Paragraph-aware windowing: split on blank lines first so a chunk rarely cuts
// a thought in half, then pack paragraphs up to the target size. Oversized
// paragraphs (minified JSON, one-line logs) fall back to hard slicing.
export function chunkText(text) {
  const normalized = String(text ?? "").replace(/\r\n/g, "\n").trim();
  if (!normalized) {
    return [];
  }
  const paragraphs = normalized.split(/\n{2,}/);
  const chunks = [];
  let buffer = "";
  let offset = 0;
  let bufferStart = 0;

  const flush = () => {
    if (!buffer.trim()) {
      buffer = "";
      return;
    }
    chunks.push({ text: buffer.trim(), start: bufferStart });
    // Carry a tail of the previous chunk forward so a sentence that straddles
    // the boundary is still retrievable from one side of it.
    const tail = buffer.slice(-CHUNK_OVERLAP_CHARS);
    buffer = tail.trim() ? `${tail.trim()}\n\n` : "";
    bufferStart = offset;
  };

  for (const paragraph of paragraphs) {
    if (paragraph.length > CHUNK_TARGET_CHARS * 2) {
      flush();
      for (let index = 0; index < paragraph.length; index += CHUNK_TARGET_CHARS) {
        chunks.push({ text: paragraph.slice(index, index + CHUNK_TARGET_CHARS), start: offset + index });
      }
      offset += paragraph.length + 2;
      bufferStart = offset;
      buffer = "";
      continue;
    }
    if (buffer.length + paragraph.length > CHUNK_TARGET_CHARS) {
      flush();
    }
    if (!buffer) {
      bufferStart = offset;
    }
    buffer += `${paragraph}\n\n`;
    offset += paragraph.length + 2;
  }
  flush();
  return chunks.map((chunk, index) => ({ ...chunk, index }));
}

export function isSupportedFilename(name) {
  const extension = path.extname(String(name ?? "")).toLowerCase();
  return SUPPORTED_EXTENSIONS.includes(extension);
}

function stripHtml(text) {
  return String(text)
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/[ \t]{2,}/g, " ");
}

export function addDocument(config, workspaceId, { name, text, source = "upload" }) {
  const cleanName = String(name ?? "document.txt").replace(/[\\/]/g, "_").slice(0, 120);
  if (!isSupportedFilename(cleanName) && source === "upload") {
    throw new Error(
      `Lizard reads text formats only. ${path.extname(cleanName) || "This file"} is not one of them — ` +
      `convert it to .txt or .md first.`,
    );
  }
  let content = String(text ?? "");
  if (content.length > MAX_DOCUMENT_CHARS) {
    content = content.slice(0, MAX_DOCUMENT_CHARS);
  }
  if (/\.(html?|xml)$/i.test(cleanName)) {
    content = stripHtml(content);
  }
  if (!content.trim()) {
    throw new Error("That file has no readable text in it.");
  }
  const chunks = chunkText(content).map((chunk) => ({
    ...chunk,
    tokens: tokenize(chunk.text),
  }));
  const document = {
    id: newId("d-"),
    workspaceId,
    name: cleanName,
    source,
    chars: content.length,
    chunkCount: chunks.length,
    addedAt: new Date().toISOString(),
    chunks,
  };
  writeDocument(config, document);
  return summarizeDocument(document);
}

function writeDocument(config, document) {
  const filePath = documentPath(config, document.workspaceId, document.id);
  ensureDirectory(path.dirname(filePath));
  fs.writeFileSync(filePath, JSON.stringify(document));
  return document;
}

function summarizeDocument(document) {
  return {
    id: document.id,
    workspaceId: document.workspaceId,
    name: document.name,
    source: document.source,
    chars: document.chars,
    chunkCount: document.chunkCount,
    addedAt: document.addedAt,
  };
}

export function listDocuments(config, workspaceId) {
  const directory = workspaceDirectory(config, workspaceId);
  if (!fs.existsSync(directory)) {
    return [];
  }
  const documents = [];
  for (const entry of fs.readdirSync(directory)) {
    if (!entry.endsWith(".json")) continue;
    try {
      const document = JSON.parse(fs.readFileSync(path.join(directory, entry), "utf8"));
      documents.push(summarizeDocument(document));
    } catch {
      // A corrupt document file should not take the whole library down.
    }
  }
  documents.sort((a, b) => String(b.addedAt).localeCompare(String(a.addedAt)));
  return documents;
}

function loadDocuments(config, workspaceId) {
  const directory = workspaceDirectory(config, workspaceId);
  if (!fs.existsSync(directory)) {
    return [];
  }
  const documents = [];
  for (const entry of fs.readdirSync(directory)) {
    if (!entry.endsWith(".json")) continue;
    try {
      documents.push(JSON.parse(fs.readFileSync(path.join(directory, entry), "utf8")));
    } catch {
      // skip unreadable document
    }
  }
  return documents;
}

export function deleteDocument(config, workspaceId, documentId) {
  const filePath = documentPath(config, workspaceId, documentId);
  if (fs.existsSync(filePath)) {
    fs.unlinkSync(filePath);
  }
  return { ok: true };
}

export function getDocumentText(config, workspaceId, documentId) {
  const filePath = documentPath(config, workspaceId, documentId);
  if (!fs.existsSync(filePath)) {
    return null;
  }
  const document = JSON.parse(fs.readFileSync(filePath, "utf8"));
  return { ...summarizeDocument(document), text: document.chunks.map((chunk) => chunk.text).join("\n\n") };
}

// ---- BM25 -----------------------------------------------------------------

const BM25_K1 = 1.5;
const BM25_B = 0.75;

/**
 * Rank workspace chunks against a query.
 * `documentIds` restricts the search to explicitly referenced documents, which
 * is what the composer's `#document` mention does.
 */
/**
 * The first chunks of documents the user named outright.
 *
 * BM25 answers "which passage matches these words", which is the wrong question
 * when the user wrote "#notes.txt tell me what is it about": after stop words
 * there are no words to match, so the search returns nothing and the model
 * reports - accurately, from where it sits - that the document was not
 * provided. Naming a document is a request for the document, so this hands over
 * its opening in reading order rather than searching inside it.
 *
 * Ordered by document, then by chunk index, so the model reads the start of the
 * file rather than an arbitrary slice of the middle.
 */
export function openingChunks(config, workspaceId, documentIds, limit = 4) {
  if (!Array.isArray(documentIds) || documentIds.length === 0) {
    return [];
  }
  const wanted = new Set(documentIds);
  const documents = loadDocuments(config, workspaceId).filter((document) => wanted.has(document.id));
  const picked = [];
  // Round-robin across the named documents so citing two does not spend the
  // whole budget on the first one.
  const perDocument = Math.max(1, Math.ceil(limit / Math.max(1, documents.length)));
  for (const document of documents) {
    const chunks = [...(document.chunks ?? [])].sort((a, b) => a.index - b.index);
    for (const chunk of chunks.slice(0, perDocument)) {
      picked.push({
        documentId: document.id,
        documentName: document.name,
        index: chunk.index,
        text: chunk.text,
        score: 0,
      });
    }
  }
  return picked.slice(0, limit);
}

export function retrieve(config, workspaceId, query, { topK = 4, documentIds = null } = {}) {
  const queryTokens = tokenize(query);
  if (queryTokens.length === 0) {
    return [];
  }
  const documents = loadDocuments(config, workspaceId).filter(
    (document) => !documentIds || documentIds.includes(document.id),
  );
  if (documents.length === 0) {
    return [];
  }

  const corpus = [];
  for (const document of documents) {
    for (const chunk of document.chunks ?? []) {
      corpus.push({
        documentId: document.id,
        documentName: document.name,
        index: chunk.index,
        text: chunk.text,
        tokens: chunk.tokens ?? tokenize(chunk.text),
      });
    }
  }
  if (corpus.length === 0) {
    return [];
  }

  const averageLength = corpus.reduce((total, chunk) => total + chunk.tokens.length, 0) / corpus.length;
  const documentFrequency = new Map();
  for (const chunk of corpus) {
    for (const token of new Set(chunk.tokens)) {
      documentFrequency.set(token, (documentFrequency.get(token) ?? 0) + 1);
    }
  }

  const scored = corpus.map((chunk) => {
    const counts = new Map();
    for (const token of chunk.tokens) {
      counts.set(token, (counts.get(token) ?? 0) + 1);
    }
    let score = 0;
    for (const token of queryTokens) {
      const frequency = counts.get(token);
      if (!frequency) continue;
      const n = documentFrequency.get(token) ?? 0;
      const idf = Math.log(1 + (corpus.length - n + 0.5) / (n + 0.5));
      const denominator = frequency + BM25_K1 * (1 - BM25_B + (BM25_B * chunk.tokens.length) / (averageLength || 1));
      score += idf * ((frequency * (BM25_K1 + 1)) / denominator);
    }
    return { ...chunk, score };
  });

  return scored
    .filter((chunk) => chunk.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, Math.max(1, topK))
    .map((chunk, rank) => ({
      rank: rank + 1,
      documentId: chunk.documentId,
      documentName: chunk.documentName,
      locator: `chunk ${chunk.index + 1}`,
      score: Number(chunk.score.toFixed(3)),
      text: chunk.text,
    }));
}

/** Renders retrieved chunks as the numbered context block the prompt carries. */
export function formatContextBlock(matches) {
  if (matches.length === 0) {
    return "";
  }
  const parts = matches.map(
    (match) => `[${match.rank}] ${match.documentName} (${match.locator})\n${match.text}`,
  );
  return [
    "Context from the user's documents:",
    "",
    parts.join("\n\n"),
    "",
    "Cite the numbered sources inline as [1], [2] when you use them.",
  ].join("\n");
}
