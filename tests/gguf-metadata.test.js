import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { activateModel, registerDownloadedModel } from "../src/lib/download-manager.js";
import { deriveLoadingPolicy, inspectGgufFile } from "../src/lib/gguf-metadata.js";

function tempConfig() {
  const runtimeDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "lizard-gguf-"));
  return {
    runtimeDirectory,
    modelDirectory: path.join(runtimeDirectory, "models"),
    catalogPath: path.join(runtimeDirectory, "model-catalog.json"),
  };
}

function encodeString(value) {
  const text = Buffer.from(String(value), "utf8");
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(text.length), 0);
  return Buffer.concat([length, text]);
}

function encodeUint32(value) {
  const buffer = Buffer.alloc(4);
  buffer.writeUInt32LE(value, 0);
  return buffer;
}

function encodeUint64(value) {
  const buffer = Buffer.alloc(8);
  buffer.writeBigUInt64LE(BigInt(value), 0);
  return buffer;
}

function kvString(key, value) {
  return Buffer.concat([encodeString(key), encodeUint32(8), encodeString(value)]);
}

function kvUint32(key, value) {
  return Buffer.concat([encodeString(key), encodeUint32(4), encodeUint32(value)]);
}

function kvStringArray(key, values) {
  const encodedValues = values.map((value) => encodeString(value));
  return Buffer.concat([
    encodeString(key),
    encodeUint32(9),
    encodeUint32(8),
    encodeUint64(values.length),
    ...encodedValues,
  ]);
}

function repeatString(value, count) {
  return Array.from({ length: count }, () => value);
}

function tensorEntry(name, dimensions, ggmlType = 0, offsetBytes = 0) {
  return Buffer.concat([
    encodeString(name),
    encodeUint32(dimensions.length),
    ...dimensions.map((dimension) => encodeUint64(dimension)),
    encodeUint32(ggmlType),
    encodeUint64(offsetBytes),
  ]);
}

function buildTensorData(size, fillByte) {
  return Buffer.alloc(size, fillByte);
}

function writeSyntheticGguf(filePath) {
  const metadata = [
    kvString("general.architecture", "llama"),
    kvUint32("general.file_type", 15),
    kvString("tokenizer.ggml.model", "llama"),
    kvUint32("llama.context_length", 4096),
    kvUint32("llama.embedding_length", 4096),
    kvUint32("llama.block_count", 2),
    kvString("tokenizer.chat_template", "{{ bos_token }}{{ prompt }}"),
    kvUint32("tokenizer.ggml.bos_token_id", 1),
    kvUint32("tokenizer.ggml.eos_token_id", 2),
    kvStringArray("tokenizer.ggml.tokens", ["<unk>", "<s>", "</s>", "hello", "world"]),
    kvStringArray("tokenizer.ggml.merges", ["h e", "he llo"]),
  ];
  const tensors = [
    tensorEntry("token_embd.weight", [4, 8], 1, 0),
    tensorEntry("output_norm.weight", [4], 1, 64),
    tensorEntry("output.weight", [8, 4], 1, 72),
    tensorEntry("blk.0.attn_norm.weight", [4], 1, 136),
    tensorEntry("blk.0.attn_q.weight", [4, 4], 1, 144),
    tensorEntry("blk.0.attn_k.weight", [4, 4], 1, 176),
    tensorEntry("blk.0.attn_v.weight", [4, 4], 1, 208),
    tensorEntry("blk.0.attn_output.weight", [4, 4], 1, 240),
    tensorEntry("blk.0.ffn_norm.weight", [4], 1, 272),
    tensorEntry("blk.0.ffn_gate.weight", [8, 4], 1, 280),
    tensorEntry("blk.0.ffn_down.weight", [4, 8], 1, 344),
    tensorEntry("blk.0.ffn_up.weight", [8, 4], 1, 408),
    tensorEntry("blk.1.attn_norm.weight", [4], 1, 472),
    tensorEntry("blk.1.attn_q.weight", [4, 4], 1, 480),
    tensorEntry("blk.1.attn_k.weight", [4, 4], 1, 512),
    tensorEntry("blk.1.attn_v.weight", [4, 4], 1, 544),
    tensorEntry("blk.1.attn_output.weight", [4, 4], 1, 576),
    tensorEntry("blk.1.ffn_norm.weight", [4], 1, 608),
    tensorEntry("blk.1.ffn_gate.weight", [8, 4], 1, 616),
    tensorEntry("blk.1.ffn_down.weight", [4, 8], 1, 680),
    tensorEntry("blk.1.ffn_up.weight", [8, 4], 1, 744),
  ];
  const header = Buffer.concat([
    Buffer.from("GGUF"),
    encodeUint32(3),
    encodeUint64(tensors.length),
    encodeUint64(metadata.length),
  ]);
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const tensorData = [
    buildTensorData(64, 0x11),
    buildTensorData(8, 0x12),
    buildTensorData(64, 0x13),
    buildTensorData(8, 0x14),
    buildTensorData(32, 0x15),
    buildTensorData(32, 0x16),
    buildTensorData(32, 0x17),
    buildTensorData(32, 0x18),
    buildTensorData(8, 0x19),
    buildTensorData(64, 0x1a),
    buildTensorData(64, 0x1b),
    buildTensorData(64, 0x1c),
    buildTensorData(8, 0x1d),
    buildTensorData(32, 0x1e),
    buildTensorData(32, 0x1f),
    buildTensorData(32, 0x20),
    buildTensorData(32, 0x21),
    buildTensorData(8, 0x22),
    buildTensorData(64, 0x23),
    buildTensorData(64, 0x24),
    buildTensorData(64, 0x25),
  ];
  const directory = Buffer.concat([header, ...metadata, ...tensors]);
  const padding = Buffer.alloc((32 - (directory.length % 32)) % 32, 0);
  fs.writeFileSync(filePath, Buffer.concat([directory, padding, ...tensorData]));
}

function writeBfloat16Gguf(filePath) {
  const metadata = [
    kvString("general.architecture", "llama"),
    kvUint32("general.file_type", 32),
    kvString("tokenizer.ggml.model", "llama"),
    kvUint32("llama.context_length", 32),
    kvUint32("llama.embedding_length", 4),
    kvUint32("llama.block_count", 0),
    kvUint32("tokenizer.ggml.bos_token_id", 1),
    kvStringArray("tokenizer.ggml.tokens", ["<unk>", "<s>", "hello", "world"]),
  ];
  const tensors = [
    tensorEntry("token_embd.weight", [4, 4], 30, 0),
    tensorEntry("output_norm.weight", [4], 30, 32),
    tensorEntry("output.weight", [4, 4], 30, 40),
  ];
  const header = Buffer.concat([
    Buffer.from("GGUF"),
    encodeUint32(3),
    encodeUint64(tensors.length),
    encodeUint64(metadata.length),
  ]);
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const tensorData = [
    buildTensorData(32, 0x11),
    buildTensorData(8, 0x12),
    buildTensorData(32, 0x13),
  ];
  const directory = Buffer.concat([header, ...metadata, ...tensors]);
  const padding = Buffer.alloc((32 - (directory.length % 32)) % 32, 0);
  fs.writeFileSync(filePath, Buffer.concat([directory, padding, ...tensorData]));
}

function writeIq4XsGguf(filePath) {
  const metadata = [
    kvString("general.architecture", "llama"),
    kvUint32("general.file_type", 30),
    kvString("tokenizer.ggml.model", "llama"),
    kvUint32("llama.context_length", 32),
    kvUint32("llama.embedding_length", 256),
    kvUint32("llama.block_count", 0),
  ];
  const tensors = [tensorEntry("output.weight", [256], 23, 0)];
  const header = Buffer.concat([
    Buffer.from("GGUF"),
    encodeUint32(3),
    encodeUint64(tensors.length),
    encodeUint64(metadata.length),
  ]);
  const directory = Buffer.concat([header, ...metadata, ...tensors]);
  const padding = Buffer.alloc((32 - (directory.length % 32)) % 32, 0);
  fs.writeFileSync(filePath, Buffer.concat([directory, padding, Buffer.alloc(136)]));
}

test("inspectGgufFile parses metadata from a valid GGUF header", () => {
  const filePath = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "lizard-gguf-parse-")), "demo.gguf");
  writeSyntheticGguf(filePath);

  const metadata = inspectGgufFile(filePath, { fallbackQuant: "Q4_K_M" });
  assert.equal(metadata.status, "ready");
  assert.equal(metadata.architecture, "llama");
  assert.equal(metadata.quant, "Q4_K_M");
  assert.equal(metadata.contextLength, 4096);
  assert.equal(metadata.embeddingLength, 4096);
  assert.equal(metadata.blockCount, 2);
  assert.equal(metadata.tensorCount, 21);
  assert.equal(metadata.tokenizer, "llama");
  assert.match(metadata.chatTemplate, /prompt/);
  assert.equal(metadata.tokenVocabularySize, 5);
  assert.deepEqual(metadata.tokenPreview, ["<unk>", "<s>", "</s>", "hello", "world"]);
  assert.equal(metadata.tokenMergeCount, 2);
  assert.deepEqual(metadata.mergePreview, ["h e", "he llo"]);
  assert.deepEqual(metadata.specialTokenTexts, { bos: "<s>", eos: "</s>" });
  assert.equal(metadata.tensorDirectoryLoaded, true);
  assert.equal(metadata.tensorDataOffsetBytes % 32, 0);
  assert.equal(metadata.tensorDataBytes, 808);
  assert.equal(metadata.tensorsPreview[0].name, "token_embd.weight");
  assert.equal(metadata.tensorsPreview[0].ggmlTypeName, "F16");
  assert.equal(metadata.tensorsPreview[0].elementCount, 32);
  assert.equal(metadata.tensorsPreview[0].byteLength, 64);
  assert.equal(Number(metadata.tensorsPreview[0].absoluteOffsetBytes), metadata.tensorDataOffsetBytes);
  assert.equal(metadata.executorShape.tensorDirectoryComplete, true);
  assert.equal(metadata.executorShape.completeLayers, 2);
  assert.deepEqual(metadata.executorShape.tensorTypes, ["F16"]);
  assert.ok(metadata.metadataHash);
});

test("inspectGgufFile recognizes BF16 tensor layouts", () => {
  const filePath = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "lizard-gguf-bf16-")), "bf16.gguf");
  writeBfloat16Gguf(filePath);

  const metadata = inspectGgufFile(filePath);
  assert.equal(metadata.status, "ready");
  assert.equal(metadata.quant, "BF16");
  assert.equal(metadata.tensorsPreview[0].ggmlTypeName, "BF16");
  assert.equal(metadata.tensorsPreview[0].byteLength, 32);
  assert.deepEqual(metadata.executorShape.tensorTypes, ["BF16"]);
});

test("inspectGgufFile recognizes IQ4_XS tensor layouts", () => {
  const filePath = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "lizard-gguf-iq4xs-")), "iq4xs.gguf");
  writeIq4XsGguf(filePath);

  const metadata = inspectGgufFile(filePath);
  assert.equal(metadata.quant, "IQ4_XS");
  assert.equal(metadata.tensorsPreview[0].ggmlTypeName, "IQ4_XS");
  assert.equal(metadata.tensorsPreview[0].byteLength, 136);
  assert.deepEqual(metadata.executorShape.tensorTypes, ["IQ4_XS"]);
});

test("inspectGgufFile falls back gracefully when metadata is truncated", () => {
  const filePath = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "lizard-gguf-short-")), "short.gguf");
  fs.writeFileSync(filePath, Buffer.from("GGUFtiny"));

  const metadata = inspectGgufFile(filePath, {
    fallbackQuant: "Q4_K_M",
    fallbackArchitecture: "llama",
  });
  assert.equal(metadata.status, "incomplete");
  assert.equal(metadata.quant, "Q4_K_M");
  assert.equal(metadata.architecture, "llama");
  assert.ok(metadata.metadataHash);
  assert.ok(metadata.warnings.length > 0);
});

test("inspectGgufFile expands the buffered metadata window for large tokenizer tables", () => {
  const filePath = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "lizard-gguf-large-")), "large.gguf");
  const largeTokens = repeatString("x".repeat(768), 7000);
  const metadata = [
    kvString("general.architecture", "llama"),
    kvUint32("general.file_type", 18),
    kvString("tokenizer.ggml.model", "gpt2"),
    kvUint32("llama.context_length", 4096),
    kvUint32("llama.embedding_length", 4096),
    kvUint32("llama.block_count", 0),
    kvUint32("tokenizer.ggml.bos_token_id", 0),
    kvStringArray("tokenizer.ggml.tokens", largeTokens),
    kvStringArray("tokenizer.ggml.merges", ["a b", "ab c"]),
  ];
  const tensors = [
    tensorEntry("token_embd.weight", [4, 4], 1, 0),
    tensorEntry("output_norm.weight", [4], 1, 32),
    tensorEntry("output.weight", [4, 4], 1, 40),
  ];
  const directory = Buffer.concat([
    Buffer.from("GGUF"),
    encodeUint32(3),
    encodeUint64(tensors.length),
    encodeUint64(metadata.length),
    ...metadata,
    ...tensors,
  ]);
  const padding = Buffer.alloc((32 - (directory.length % 32)) % 32, 0);
  const tensorData = Buffer.concat([
    buildTensorData(32, 0x11),
    buildTensorData(8, 0x12),
    buildTensorData(32, 0x13),
  ]);
  fs.writeFileSync(filePath, Buffer.concat([directory, padding, tensorData]));

  const parsed = inspectGgufFile(filePath, { fallbackQuant: "Q6_K" });
  assert.equal(parsed.status, "ready");
  assert.equal(parsed.tokenizer, "gpt2");
  assert.equal(parsed.quant, "Q6_K");
  assert.equal(parsed.tokenVocabularySize, largeTokens.length);
  assert.equal(parsed.tokenMergeCount, 2);
  assert.equal(parsed.raw["tokenizer.ggml.tokens"].length, largeTokens.length);
  assert.equal(parsed.raw["tokenizer.ggml.merges"][0], "a b");
});

test("inspectGgufFile matches current llama.cpp file type quant enums", () => {
  const q4Path = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "lizard-gguf-q4km-")), "q4.gguf");
  writeSyntheticGguf(q4Path);
  const q4 = inspectGgufFile(q4Path);
  assert.equal(q4.quant, "Q4_K_M");

  const q6Path = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "lizard-gguf-q6k-")), "q6.gguf");
  const metadata = [
    kvString("general.architecture", "llama"),
    kvUint32("general.file_type", 18),
    kvString("tokenizer.ggml.model", "llama"),
    kvUint32("llama.context_length", 32),
    kvUint32("llama.embedding_length", 32),
    kvUint32("llama.block_count", 0),
  ];
  const tensors = [
    tensorEntry("token_embd.weight", [4, 4], 1, 0),
    tensorEntry("output_norm.weight", [4], 1, 32),
    tensorEntry("output.weight", [4, 4], 1, 40),
  ];
  const directory = Buffer.concat([
    Buffer.from("GGUF"),
    encodeUint32(3),
    encodeUint64(tensors.length),
    encodeUint64(metadata.length),
    ...metadata,
    ...tensors,
  ]);
  const padding = Buffer.alloc((32 - (directory.length % 32)) % 32, 0);
  const tensorData = Buffer.concat([
    buildTensorData(32, 0x11),
    buildTensorData(8, 0x12),
    buildTensorData(32, 0x13),
  ]);
  fs.writeFileSync(q6Path, Buffer.concat([directory, padding, tensorData]));
  const q6 = inspectGgufFile(q6Path);
  assert.equal(q6.quant, "Q6_K");
});

test("deriveLoadingPolicy uses GGUF metadata and hardware headroom", () => {
  const policy = deriveLoadingPolicy({
    gguf: {
      status: "ready",
      contextLength: 8192,
      fileSizeGb: 2.5,
      warnings: [],
    },
    hardware: {
      memory: { plannerBudgetGB: 16, availableGB: 16 },
      primaryGpu: { vramGB: 8 },
    },
    estimate: { ramRequiredGB: 10 },
    requestedContextSize: 12000,
    requestedGpuLayers: "auto",
    mode: "pro",
  });

  assert.equal(policy.warmupPrompt, true);
  assert.equal(policy.expectedLoadClass, "medium");
  assert.equal(policy.memoryRisk, "low");
  assert.equal(policy.recommendedContextSize, 8192);
  assert.equal(policy.recommendedGpuLayers, "auto");
});

test("registerDownloadedModel and activateModel persist GGUF metadata", () => {
  const config = tempConfig();
  const filePath = path.join(config.modelDirectory, "demo.gguf");
  writeSyntheticGguf(filePath);
  const gguf = inspectGgufFile(filePath, { fallbackQuant: "Q4_K_M" });

  const record = registerDownloadedModel({
    config,
    entry: {
      id: "demo-model",
      source: "huggingface",
      hfRepo: "demo/repo",
      filename: "demo.gguf",
      importedPath: filePath,
      quant: "Q4_K_M",
      gguf,
      sha256: "abc123",
    },
  });
  assert.equal(record.gguf.metadataHash, gguf.metadataHash);

  const active = activateModel({ config, modelId: "demo-model" });
  assert.equal(active.gguf.metadataHash, gguf.metadataHash);
  assert.equal(active.gguf.architecture, "llama");
});
