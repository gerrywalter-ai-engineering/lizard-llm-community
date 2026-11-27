import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { inspectGgufFile } from "../src/lib/gguf-metadata.js";
import { NativeWorkerProcessClient } from "../src/runtime/worker/process-client.js";

function tempDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "lizard-native-compiled-"));
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

function encodeFloat32(value) {
  const buffer = Buffer.alloc(4);
  buffer.writeFloatLE(value, 0);
  return buffer;
}

function encodeFloat16(value) {
  const sign = value < 0 ? 0x8000 : 0;
  const absolute = Math.abs(value);
  if (absolute === 0) {
    const buffer = Buffer.alloc(2);
    buffer.writeUInt16LE(sign, 0);
    return buffer;
  }

  const exponent = Math.floor(Math.log2(absolute));
  const normalized = absolute / (2 ** exponent);
  const halfExponent = exponent + 15;
  const mantissa = Math.round((normalized - 1) * 1024);
  const bits = sign | ((halfExponent & 0x1f) << 10) | (mantissa & 0x3ff);
  const buffer = Buffer.alloc(2);
  buffer.writeUInt16LE(bits, 0);
  return buffer;
}

function encodeBfloat16(value) {
  const buffer = Buffer.alloc(4);
  buffer.writeFloatLE(value, 0);
  return buffer.subarray(2, 4);
}

function kvString(key, value) {
  return Buffer.concat([encodeString(key), encodeUint32(8), encodeString(value)]);
}

function kvUint32(key, value) {
  return Buffer.concat([encodeString(key), encodeUint32(4), encodeUint32(value)]);
}

function kvStringArray(key, values) {
  return Buffer.concat([
    encodeString(key),
    encodeUint32(9),
    encodeUint32(8),
    encodeUint64(values.length),
    ...values.map((value) => encodeString(value)),
  ]);
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

function encodeFloatTensor(values) {
  return Buffer.concat(values.map((value) => encodeFloat32(value)));
}

function encodeBfloat16Tensor(values) {
  return Buffer.concat(values.map((value) => encodeBfloat16(value)));
}

function encodeQ4KScales(scales, mins) {
  const bytes = Buffer.alloc(12, 0);
  for (let index = 0; index < 4; index += 1) {
    bytes[index] = (scales[index] & 0x3f) | (((scales[index + 4] >> 4) & 0x03) << 6);
    bytes[index + 4] = (mins[index] & 0x3f) | (((mins[index + 4] >> 4) & 0x03) << 6);
    bytes[index + 8] = (scales[index + 4] & 0x0f) | ((mins[index + 4] & 0x0f) << 4);
  }
  return bytes;
}

function encodeQ4KTensor(values) {
  const blockSize = 256;
  const padded = [...values];
  while (padded.length % blockSize !== 0) {
    padded.push(0);
  }

  const blocks = [];
  for (let offset = 0; offset < padded.length; offset += blockSize) {
    const block = padded.slice(offset, offset + blockSize);
    const scales = Array.from({ length: 8 }, () => 1);
    const mins = Array.from({ length: 8 }, () => 0);
    const qs = Buffer.alloc(blockSize / 2, 0);
    for (let chunk = 0; chunk < blockSize; chunk += 64) {
      for (let item = 0; item < 32; item += 1) {
        const low = Math.max(0, Math.min(15, Math.round(block[chunk + item] ?? 0)));
        const high = Math.max(0, Math.min(15, Math.round(block[chunk + 32 + item] ?? 0)));
        qs[(chunk / 2) + item] = low | (high << 4);
      }
    }
    blocks.push(Buffer.concat([
      encodeFloat16(1),
      encodeFloat16(0),
      encodeQ4KScales(scales, mins),
      qs,
    ]));
  }

  return Buffer.concat(blocks);
}

function encodeQ5KTensor(values) {
  const blockSize = 256;
  const padded = [...values];
  while (padded.length % blockSize !== 0) {
    padded.push(0);
  }

  const blocks = [];
  for (let offset = 0; offset < padded.length; offset += blockSize) {
    const block = padded.slice(offset, offset + blockSize);
    const scales = Array.from({ length: 8 }, () => 1);
    const mins = Array.from({ length: 8 }, () => 0);
    const qh = Buffer.alloc(blockSize / 8, 0);
    const qs = Buffer.alloc(blockSize / 2, 0);
    const bitMasks = [
      [1, 2],
      [4, 8],
      [16, 32],
      [64, 128],
    ];

    for (let chunk = 0; chunk < blockSize; chunk += 64) {
      const chunkIndex = chunk / 64;
      const [lowMask, highMask] = bitMasks[chunkIndex];
      for (let item = 0; item < 32; item += 1) {
        const low = Math.max(0, Math.min(31, Math.round(block[chunk + item] ?? 0)));
        const high = Math.max(0, Math.min(31, Math.round(block[chunk + 32 + item] ?? 0)));
        qs[(chunk / 2) + item] = (low & 0x0f) | ((high & 0x0f) << 4);
        if ((low & 0x10) !== 0) {
          qh[item] |= lowMask;
        }
        if ((high & 0x10) !== 0) {
          qh[item] |= highMask;
        }
      }
    }

    blocks.push(Buffer.concat([
      encodeFloat16(1),
      encodeFloat16(0),
      encodeQ4KScales(scales, mins),
      qh,
      qs,
    ]));
  }

  return Buffer.concat(blocks);
}

function encodeQ6KTensor(values) {
  const blockSize = 256;
  const padded = [...values];
  while (padded.length % blockSize !== 0) {
    padded.push(0);
  }

  const blocks = [];
  for (let offset = 0; offset < padded.length; offset += blockSize) {
    const block = padded.slice(offset, offset + blockSize);
    const ql = Buffer.alloc(blockSize / 2, 0);
    const qh = Buffer.alloc(blockSize / 4, 0);
    const scales = Buffer.alloc(blockSize / 16, 1);

    for (let chunk = 0; chunk < blockSize; chunk += 128) {
      const qlBase = chunk / 2;
      const qhBase = chunk / 4;
      for (let item = 0; item < 32; item += 1) {
        const encode = (value) => {
          const quantized = Math.max(-32, Math.min(31, Math.round(value ?? 0)));
          return quantized + 32;
        };
        const q1 = encode(block[chunk + item]);
        const q2 = encode(block[chunk + 32 + item]);
        const q3 = encode(block[chunk + 64 + item]);
        const q4 = encode(block[chunk + 96 + item]);

        ql[qlBase + item] = (q1 & 0x0f) | ((q3 & 0x0f) << 4);
        ql[qlBase + 32 + item] = (q2 & 0x0f) | ((q4 & 0x0f) << 4);
        qh[qhBase + item] = ((q1 >> 4) & 0x03)
          | (((q2 >> 4) & 0x03) << 2)
          | (((q3 >> 4) & 0x03) << 4)
          | (((q4 >> 4) & 0x03) << 6);
      }
    }

    blocks.push(Buffer.concat([
      ql,
      qh,
      scales,
      encodeFloat16(1),
    ]));
  }

  return Buffer.concat(blocks);
}

// K-quants are defined in 256-value superblocks (QK_K), so a legal K-quant
// tensor's first dimension must be a whole multiple of 256. ggml rejects
// anything else, and no real quantizer emits it - a Q4_K tensor with ne0=2
// cannot exist in a GGUF a user could actually download. These fixtures stay
// logically two-dimensional: every row carries its meaningful leading values
// and is widened with zeros to one full superblock, so the file describes a
// model that could really exist while the arithmetic below is unchanged.
const K_QUANT_BLOCK = 256;

function kQuantRows(rows) {
  const values = [];
  for (const row of rows) {
    for (let index = 0; index < K_QUANT_BLOCK; index += 1) {
      values.push(row[index] ?? 0);
    }
  }
  return values;
}

// A square [256, 256] projection whose only non-zero entries live in the
// leading 2x2 corner, which is the sub-space these fixtures actually use.
function kQuantSquare(corner) {
  return kQuantRows(
    Array.from({ length: K_QUANT_BLOCK }, (unused, index) => corner[index] ?? []),
  );
}

function writeZeroLayerGguf(filePath) {
  const metadata = [
    kvString("general.architecture", "llama"),
    kvUint32("general.file_type", 0),
    kvString("tokenizer.ggml.model", "synthetic-greedy"),
    kvUint32("llama.context_length", 32),
    kvUint32("llama.embedding_length", 2),
    kvUint32("llama.block_count", 0),
    kvString("tokenizer.chat_template", "{{ prompt }}"),
    kvUint32("tokenizer.ggml.bos_token_id", 0),
    kvUint32("tokenizer.ggml.eos_token_id", 4),
    kvStringArray("tokenizer.ggml.tokens", ["<s>", "hello", " ", "world", "</s>"]),
  ];

  const tokenEmbeddings = encodeFloatTensor([
    0, 0,
    1, 0,
    0, 1,
    1, 1,
    -1, 0,
  ]);
  const outputNorm = encodeFloatTensor([1, 1]);
  const outputWeights = encodeFloatTensor([
    0, 0,
    0, 0,
    0, 0,
    1.5, 0,
    1, 1,
  ]);

  const tensors = [
    tensorEntry("token_embd.weight", [2, 5], 0, 0),
    tensorEntry("output_norm.weight", [2], 0, tokenEmbeddings.length),
    tensorEntry("output.weight", [2, 5], 0, tokenEmbeddings.length + outputNorm.length),
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
  const tensorData = Buffer.concat([tokenEmbeddings, outputNorm, outputWeights]);

  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, Buffer.concat([directory, padding, tensorData]));
}

function writeGpt2BpeZeroLayerGguf(filePath) {
  const metadata = [
    kvString("general.architecture", "llama"),
    kvUint32("general.file_type", 0),
    kvString("tokenizer.ggml.model", "gpt2"),
    kvUint32("llama.context_length", 32),
    kvUint32("llama.embedding_length", 2),
    kvUint32("llama.block_count", 0),
    kvString("tokenizer.chat_template", "{{ prompt }}"),
    kvUint32("tokenizer.ggml.bos_token_id", 0),
    kvUint32("tokenizer.ggml.eos_token_id", 10),
    kvStringArray("tokenizer.ggml.tokens", ["<s>", "H", "e", "l", "o", "He", "ll", "Hell", "Hello", "\u0120Hello", "</s>"]),
    kvStringArray("tokenizer.ggml.merges", ["H e", "l l", "He ll", "Hell o", "\u0120 Hello"]),
  ];

  const tokenEmbeddings = encodeFloatTensor([
    0, 0,
    0.1, 0,
    0.2, 0,
    0.3, 0,
    0.4, 0,
    0.5, 0,
    0.6, 0,
    0.7, 0,
    0.8, 0,
    0.9, 0,
    -1, 0,
  ]);
  const outputNorm = encodeFloatTensor([1, 1]);
  const outputWeights = encodeFloatTensor([
    0, 0,
    0, 0,
    0, 0,
    0, 0,
    0, 0,
    0, 0,
    0, 0,
    0, 0,
    1.5, 0,
    1.2, 0,
    1, 1,
  ]);

  const tensors = [
    tensorEntry("token_embd.weight", [2, 11], 0, 0),
    tensorEntry("output_norm.weight", [2], 0, tokenEmbeddings.length),
    tensorEntry("output.weight", [2, 11], 0, tokenEmbeddings.length + outputNorm.length),
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
  const tensorData = Buffer.concat([tokenEmbeddings, outputNorm, outputWeights]);

  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, Buffer.concat([directory, padding, tensorData]));
}

function writeUtf8Gpt2BpeZeroLayerGguf(filePath) {
  const metadata = [
    kvString("general.architecture", "llama"),
    kvUint32("general.file_type", 0),
    kvString("tokenizer.ggml.model", "gpt2"),
    kvUint32("llama.context_length", 32),
    kvUint32("llama.embedding_length", 2),
    kvUint32("llama.block_count", 0),
    kvString("tokenizer.chat_template", "{{ prompt }}"),
    kvUint32("tokenizer.ggml.bos_token_id", 0),
    kvUint32("tokenizer.ggml.eos_token_id", 5),
    kvStringArray("tokenizer.ggml.tokens", ["<s>", "Ã", "©", "Ã©", "ĠÃ©", "</s>"]),
    kvStringArray("tokenizer.ggml.merges", ["Ã ©", "Ġ Ã©"]),
  ];

  const tokenEmbeddings = encodeFloatTensor([
    0, 0,
    0.1, 0,
    0.2, 0,
    0.3, 0,
    0.4, 0,
    -1, 0,
  ]);
  const outputNorm = encodeFloatTensor([1, 1]);
  const outputWeights = encodeFloatTensor([
    0, 0,
    0, 0,
    0, 0,
    0, 0,
    1.5, 0,
    1, 1,
  ]);

  const tensors = [
    tensorEntry("token_embd.weight", [2, 6], 0, 0),
    tensorEntry("output_norm.weight", [2], 0, tokenEmbeddings.length),
    tensorEntry("output.weight", [2, 6], 0, tokenEmbeddings.length + outputNorm.length),
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
  const tensorData = Buffer.concat([tokenEmbeddings, outputNorm, outputWeights]);

  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, Buffer.concat([directory, padding, tensorData]));
}

function writeBfloat16ZeroLayerGguf(filePath) {
  const metadata = [
    kvString("general.architecture", "llama"),
    kvUint32("general.file_type", 32),
    kvString("tokenizer.ggml.model", "synthetic-greedy"),
    kvUint32("llama.context_length", 32),
    kvUint32("llama.embedding_length", 2),
    kvUint32("llama.block_count", 0),
    kvString("tokenizer.chat_template", "{{ prompt }}"),
    kvUint32("tokenizer.ggml.bos_token_id", 0),
    kvUint32("tokenizer.ggml.eos_token_id", 4),
    kvStringArray("tokenizer.ggml.tokens", ["<s>", "hello", " ", "world", "</s>"]),
  ];

  const tokenEmbeddings = encodeBfloat16Tensor([
    0, 0,
    1, 0,
    0, 1,
    1, 1,
    -1, 0,
  ]);
  const outputNorm = encodeBfloat16Tensor([1, 1]);
  const outputWeights = encodeBfloat16Tensor([
    0, 0,
    0, 0,
    0, 0,
    1.5, 0,
    1, 1,
  ]);

  const tensors = [
    tensorEntry("token_embd.weight", [2, 5], 30, 0),
    tensorEntry("output_norm.weight", [2], 30, tokenEmbeddings.length),
    tensorEntry("output.weight", [2, 5], 30, tokenEmbeddings.length + outputNorm.length),
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
  const tensorData = Buffer.concat([tokenEmbeddings, outputNorm, outputWeights]);

  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, Buffer.concat([directory, padding, tensorData]));
}

function writeTokenizerlessZeroLayerGguf(filePath) {
  const metadata = [
    kvString("general.architecture", "llama"),
    kvUint32("general.file_type", 0),
    kvString("tokenizer.ggml.model", "synthetic-greedy"),
    kvUint32("llama.context_length", 32),
    kvUint32("llama.embedding_length", 2),
    kvUint32("llama.block_count", 0),
    kvString("tokenizer.chat_template", "{{ prompt }}"),
    kvUint32("tokenizer.ggml.bos_token_id", 0),
    kvUint32("tokenizer.ggml.eos_token_id", 4),
  ];

  const tokenEmbeddings = encodeFloatTensor([
    0, 0,
    1, 0,
    0, 1,
    1, 1,
    -1, 0,
  ]);
  const outputNorm = encodeFloatTensor([1, 1]);
  const outputWeights = encodeFloatTensor([
    0, 0,
    0, 0,
    0, 0,
    1.5, 0,
    1, 1,
  ]);

  const tensors = [
    tensorEntry("token_embd.weight", [2, 5], 0, 0),
    tensorEntry("output_norm.weight", [2], 0, tokenEmbeddings.length),
    tensorEntry("output.weight", [2, 5], 0, tokenEmbeddings.length + outputNorm.length),
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
  const tensorData = Buffer.concat([tokenEmbeddings, outputNorm, outputWeights]);

  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, Buffer.concat([directory, padding, tensorData]));
}

function writeOneLayerGguf(filePath) {
  const metadata = [
    kvString("general.architecture", "llama"),
    kvUint32("general.file_type", 0),
    kvString("tokenizer.ggml.model", "synthetic-greedy"),
    kvUint32("llama.context_length", 32),
    kvUint32("llama.embedding_length", 2),
    kvUint32("llama.block_count", 1),
    kvUint32("llama.attention.head_count", 1),
    kvUint32("llama.attention.head_count_kv", 1),
    kvString("tokenizer.chat_template", "{{ prompt }}"),
    kvUint32("tokenizer.ggml.bos_token_id", 0),
    kvUint32("tokenizer.ggml.eos_token_id", 4),
    kvStringArray("tokenizer.ggml.tokens", ["<s>", "hello", " ", "world", "</s>"]),
  ];

  const tokenEmbeddings = encodeFloatTensor([
    0, 0,
    1, 0,
    0, 1,
    1, 1,
    -1, 0,
  ]);
  const vectorOnes = encodeFloatTensor([1, 1]);
  const zeros2x2 = encodeFloatTensor([
    0, 0,
    0, 0,
  ]);
  const identity2x2 = encodeFloatTensor([
    1, 0,
    0, 1,
  ]);
  const outputWeights = encodeFloatTensor([
    0, 0,
    0, 0,
    0, 0,
    1.5, 0,
    1, 1,
  ]);

  const tensors = [];
  const tensorData = [];
  let offsetBytes = 0;
  const pushTensor = (name, dimensions, payload, ggmlType = 0) => {
    tensors.push(tensorEntry(name, dimensions, ggmlType, offsetBytes));
    tensorData.push(payload);
    offsetBytes += payload.length;
  };

  // F32 tensors have no superblock constraint, so this fixture stays at the
  // original two-wide shape - only the K-quant fixtures need full 256 rows.
  pushTensor("token_embd.weight", [2, 5], tokenEmbeddings);
  pushTensor("blk.0.attn_norm.weight", [2], vectorOnes);
  pushTensor("blk.0.attn_q.weight", [2, 2], zeros2x2);
  pushTensor("blk.0.attn_k.weight", [2, 2], zeros2x2);
  pushTensor("blk.0.attn_v.weight", [2, 2], identity2x2);
  pushTensor("blk.0.attn_output.weight", [2, 2], zeros2x2);
  pushTensor("blk.0.ffn_norm.weight", [2], vectorOnes);
  pushTensor("blk.0.ffn_gate.weight", [2, 2], zeros2x2);
  pushTensor("blk.0.ffn_down.weight", [2, 2], zeros2x2);
  pushTensor("blk.0.ffn_up.weight", [2, 2], zeros2x2);
  pushTensor("output_norm.weight", [2], vectorOnes);
  pushTensor("output.weight", [2, 5], outputWeights);

  const directory = Buffer.concat([
    Buffer.from("GGUF"),
    encodeUint32(3),
    encodeUint64(tensors.length),
    encodeUint64(metadata.length),
    ...metadata,
    ...tensors,
  ]);
  const padding = Buffer.alloc((32 - (directory.length % 32)) % 32, 0);

  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, Buffer.concat([directory, padding, ...tensorData]));
}

function writeGroupedQueryAttentionGguf(filePath) {
  const metadata = [
    kvString("general.architecture", "llama"),
    kvUint32("general.file_type", 0),
    kvString("tokenizer.ggml.model", "synthetic-gqa"),
    kvUint32("llama.context_length", 32),
    kvUint32("llama.embedding_length", 4),
    kvUint32("llama.block_count", 1),
    kvUint32("llama.attention.head_count", 2),
    kvUint32("llama.attention.head_count_kv", 1),
    kvString("tokenizer.chat_template", "{{ prompt }}"),
    kvUint32("tokenizer.ggml.bos_token_id", 0),
    kvUint32("tokenizer.ggml.eos_token_id", 4),
    kvStringArray("tokenizer.ggml.tokens", ["<s>", "hello", " ", "world", "</s>"]),
  ];

  const tokenEmbeddings = encodeFloatTensor([
    0, 0, 0, 0,
    2, 0, 0, 0,
    0, 2, 0, 0,
    0, 0, 0, 0,
    -1, 0, 0, 0,
  ]);
  const vectorOnes = encodeFloatTensor([1, 1, 1, 1]);
  const zeroMatrix4x4 = encodeFloatTensor([
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
  ]);
  const valueProjection = encodeFloatTensor([
    1, 0, 0, 0,
    0, 1, 0, 0,
  ]);
  const attentionOutput = encodeFloatTensor([
    1, 0, 0, 0,
    0, 1, 0, 0,
    1, 0, 0, 0,
    0, 1, 0, 0,
  ]);
  const zeroMatrix4x8 = encodeFloatTensor(new Array(32).fill(0));
  const zeroMatrix8x4 = encodeFloatTensor(new Array(32).fill(0));
  const outputWeights = encodeFloatTensor([
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 1, 0, 0,
    0, 0.5, 0, 0,
  ]);

  const tensors = [];
  const tensorData = [];
  let offsetBytes = 0;
  const pushTensor = (name, dimensions, payload, ggmlType = 0) => {
    tensors.push(tensorEntry(name, dimensions, ggmlType, offsetBytes));
    tensorData.push(payload);
    offsetBytes += payload.length;
  };

  pushTensor("token_embd.weight", [4, 5], tokenEmbeddings);
  pushTensor("blk.0.attn_norm.weight", [4], vectorOnes);
  pushTensor("blk.0.attn_q.weight", [4, 4], zeroMatrix4x4);
  pushTensor("blk.0.attn_k.weight", [4, 2], valueProjection);
  pushTensor("blk.0.attn_v.weight", [4, 2], valueProjection);
  pushTensor("blk.0.attn_output.weight", [4, 4], attentionOutput);
  pushTensor("blk.0.ffn_norm.weight", [4], vectorOnes);
  pushTensor("blk.0.ffn_gate.weight", [4, 8], zeroMatrix4x8);
  pushTensor("blk.0.ffn_down.weight", [8, 4], zeroMatrix8x4);
  pushTensor("blk.0.ffn_up.weight", [4, 8], zeroMatrix4x8);
  pushTensor("output_norm.weight", [4], vectorOnes);
  pushTensor("output.weight", [4, 5], outputWeights);

  const directory = Buffer.concat([
    Buffer.from("GGUF"),
    encodeUint32(3),
    encodeUint64(tensors.length),
    encodeUint64(metadata.length),
    ...metadata,
    ...tensors,
  ]);
  const padding = Buffer.alloc((32 - (directory.length % 32)) % 32, 0);

  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, Buffer.concat([directory, padding, ...tensorData]));
}

function writeSamplingZeroLayerGguf(filePath) {
  const metadata = [
    kvString("general.architecture", "llama"),
    kvUint32("general.file_type", 0),
    kvString("tokenizer.ggml.model", "synthetic-sampling"),
    kvUint32("llama.context_length", 32),
    kvUint32("llama.embedding_length", 2),
    kvUint32("llama.block_count", 0),
    kvString("tokenizer.chat_template", "{{ prompt }}"),
    kvUint32("tokenizer.ggml.bos_token_id", 0),
    kvUint32("tokenizer.ggml.eos_token_id", 4),
    kvStringArray("tokenizer.ggml.tokens", ["<s>", "hello", " ", "world", "</s>"]),
  ];

  const tokenEmbeddings = encodeFloatTensor([
    0, 0,
    1, 0,
    0, 1,
    1, 1,
    -1, 0,
  ]);
  const outputNorm = encodeFloatTensor([1, 1]);
  const outputWeights = encodeFloatTensor([
    0, 0,
    0, 0,
    0, 0,
    1, 0,
    1, 0,
  ]);

  const tensors = [
    tensorEntry("token_embd.weight", [2, 5], 0, 0),
    tensorEntry("output_norm.weight", [2], 0, tokenEmbeddings.length),
    tensorEntry("output.weight", [2, 5], 0, tokenEmbeddings.length + outputNorm.length),
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
  const tensorData = Buffer.concat([tokenEmbeddings, outputNorm, outputWeights]);

  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, Buffer.concat([directory, padding, tensorData]));
}

function writeQ4KOneLayerGguf(filePath) {
  const metadata = [
    kvString("general.architecture", "llama"),
    kvUint32("general.file_type", 15),
    kvString("tokenizer.ggml.model", "synthetic-q4-k"),
    kvUint32("llama.context_length", 32),
    kvUint32("llama.embedding_length", K_QUANT_BLOCK),
    kvUint32("llama.block_count", 1),
    kvUint32("llama.attention.head_count", 1),
    kvUint32("llama.attention.head_count_kv", 1),
    kvString("tokenizer.chat_template", "{{ prompt }}"),
    kvUint32("tokenizer.ggml.bos_token_id", 0),
    kvUint32("tokenizer.ggml.eos_token_id", 4),
    kvStringArray("tokenizer.ggml.tokens", ["<s>", "hello", " ", "world", "</s>"]),
  ];

  const tokenEmbeddings = encodeQ4KTensor(kQuantRows([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0],
  ]));
  const vectorOnes = encodeQ4KTensor(kQuantRows([[1, 1]]));
  const zeros2x2 = encodeQ4KTensor(kQuantSquare([]));
  const identity2x2 = encodeQ4KTensor(kQuantSquare([[1, 0], [0, 1]]));
  const outputWeights = encodeQ4KTensor(kQuantRows([
    [0, 0],
    [0, 0],
    [0, 0],
    [2, 0],
    [1, 2],
  ]));

  const tensors = [];
  const tensorData = [];
  let offsetBytes = 0;
  const pushTensor = (name, dimensions, payload, ggmlType = 12) => {
    tensors.push(tensorEntry(name, dimensions, ggmlType, offsetBytes));
    tensorData.push(payload);
    offsetBytes += payload.length;
  };

  pushTensor("token_embd.weight", [K_QUANT_BLOCK, 5], tokenEmbeddings);
  pushTensor("blk.0.attn_norm.weight", [K_QUANT_BLOCK], vectorOnes);
  pushTensor("blk.0.attn_q.weight", [K_QUANT_BLOCK, K_QUANT_BLOCK], zeros2x2);
  pushTensor("blk.0.attn_k.weight", [K_QUANT_BLOCK, K_QUANT_BLOCK], zeros2x2);
  pushTensor("blk.0.attn_v.weight", [K_QUANT_BLOCK, K_QUANT_BLOCK], identity2x2);
  pushTensor("blk.0.attn_output.weight", [K_QUANT_BLOCK, K_QUANT_BLOCK], zeros2x2);
  pushTensor("blk.0.ffn_norm.weight", [K_QUANT_BLOCK], vectorOnes);
  pushTensor("blk.0.ffn_gate.weight", [K_QUANT_BLOCK, K_QUANT_BLOCK], zeros2x2);
  pushTensor("blk.0.ffn_down.weight", [K_QUANT_BLOCK, K_QUANT_BLOCK], zeros2x2);
  pushTensor("blk.0.ffn_up.weight", [K_QUANT_BLOCK, K_QUANT_BLOCK], zeros2x2);
  pushTensor("output_norm.weight", [K_QUANT_BLOCK], vectorOnes);
  pushTensor("output.weight", [K_QUANT_BLOCK, 5], outputWeights);

  const directory = Buffer.concat([
    Buffer.from("GGUF"),
    encodeUint32(3),
    encodeUint64(tensors.length),
    encodeUint64(metadata.length),
    ...metadata,
    ...tensors,
  ]);
  const padding = Buffer.alloc((32 - (directory.length % 32)) % 32, 0);

  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, Buffer.concat([directory, padding, ...tensorData]));
}

function writeQ5KOneLayerGguf(filePath) {
  const metadata = [
    kvString("general.architecture", "llama"),
    kvUint32("general.file_type", 17),
    kvString("tokenizer.ggml.model", "synthetic-q5-k"),
    kvUint32("llama.context_length", 32),
    kvUint32("llama.embedding_length", K_QUANT_BLOCK),
    kvUint32("llama.block_count", 1),
    kvUint32("llama.attention.head_count", 1),
    kvUint32("llama.attention.head_count_kv", 1),
    kvString("tokenizer.chat_template", "{{ prompt }}"),
    kvUint32("tokenizer.ggml.bos_token_id", 0),
    kvUint32("tokenizer.ggml.eos_token_id", 4),
    kvStringArray("tokenizer.ggml.tokens", ["<s>", "hello", " ", "world", "</s>"]),
  ];

  const tokenEmbeddings = encodeQ5KTensor(kQuantRows([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0],
  ]));
  const vectorOnes = encodeQ5KTensor(kQuantRows([[1, 1]]));
  const zeros2x2 = encodeQ5KTensor(kQuantSquare([]));
  const identity2x2 = encodeQ5KTensor(kQuantSquare([[1, 0], [0, 1]]));
  const outputWeights = encodeQ5KTensor(kQuantRows([
    [0, 0],
    [0, 0],
    [0, 0],
    [2, 0],
    [1, 2],
  ]));

  const tensors = [];
  const tensorData = [];
  let offsetBytes = 0;
  const pushTensor = (name, dimensions, payload, ggmlType = 13) => {
    tensors.push(tensorEntry(name, dimensions, ggmlType, offsetBytes));
    tensorData.push(payload);
    offsetBytes += payload.length;
  };

  pushTensor("token_embd.weight", [K_QUANT_BLOCK, 5], tokenEmbeddings);
  pushTensor("blk.0.attn_norm.weight", [K_QUANT_BLOCK], vectorOnes);
  pushTensor("blk.0.attn_q.weight", [K_QUANT_BLOCK, K_QUANT_BLOCK], zeros2x2);
  pushTensor("blk.0.attn_k.weight", [K_QUANT_BLOCK, K_QUANT_BLOCK], zeros2x2);
  pushTensor("blk.0.attn_v.weight", [K_QUANT_BLOCK, K_QUANT_BLOCK], identity2x2);
  pushTensor("blk.0.attn_output.weight", [K_QUANT_BLOCK, K_QUANT_BLOCK], zeros2x2);
  pushTensor("blk.0.ffn_norm.weight", [K_QUANT_BLOCK], vectorOnes);
  pushTensor("blk.0.ffn_gate.weight", [K_QUANT_BLOCK, K_QUANT_BLOCK], zeros2x2);
  pushTensor("blk.0.ffn_down.weight", [K_QUANT_BLOCK, K_QUANT_BLOCK], zeros2x2);
  pushTensor("blk.0.ffn_up.weight", [K_QUANT_BLOCK, K_QUANT_BLOCK], zeros2x2);
  pushTensor("output_norm.weight", [K_QUANT_BLOCK], vectorOnes);
  pushTensor("output.weight", [K_QUANT_BLOCK, 5], outputWeights);

  const directory = Buffer.concat([
    Buffer.from("GGUF"),
    encodeUint32(3),
    encodeUint64(tensors.length),
    encodeUint64(metadata.length),
    ...metadata,
    ...tensors,
  ]);
  const padding = Buffer.alloc((32 - (directory.length % 32)) % 32, 0);

  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, Buffer.concat([directory, padding, ...tensorData]));
}

function writeMixedQ4Q6OneLayerGguf(filePath) {
  const metadata = [
    kvString("general.architecture", "llama"),
    kvUint32("general.file_type", 15),
    kvString("tokenizer.ggml.model", "synthetic-q4-q6-k"),
    kvUint32("llama.context_length", 32),
    kvUint32("llama.embedding_length", K_QUANT_BLOCK),
    kvUint32("llama.block_count", 1),
    kvUint32("llama.attention.head_count", 1),
    kvUint32("llama.attention.head_count_kv", 1),
    kvString("tokenizer.chat_template", "{{ prompt }}"),
    kvUint32("tokenizer.ggml.bos_token_id", 0),
    kvUint32("tokenizer.ggml.eos_token_id", 4),
    kvStringArray("tokenizer.ggml.tokens", ["<s>", "hello", " ", "world", "</s>"]),
  ];

  const tokenEmbeddings = encodeQ6KTensor(kQuantRows([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0],
  ]));
  const vectorOnes = encodeQ4KTensor(kQuantRows([[1, 1]]));
  const zeros2x2 = encodeQ4KTensor(kQuantSquare([]));
  const identity2x2 = encodeQ6KTensor(kQuantSquare([[1, 0], [0, 1]]));
  const outputWeights = encodeQ6KTensor(kQuantRows([
    [0, 0],
    [0, 0],
    [0, 0],
    [2, 0],
    [1, 2],
  ]));

  const tensors = [];
  const tensorData = [];
  let offsetBytes = 0;
  const pushTensor = (name, dimensions, payload, ggmlType) => {
    tensors.push(tensorEntry(name, dimensions, ggmlType, offsetBytes));
    tensorData.push(payload);
    offsetBytes += payload.length;
  };

  pushTensor("token_embd.weight", [K_QUANT_BLOCK, 5], tokenEmbeddings, 14);
  pushTensor("blk.0.attn_norm.weight", [K_QUANT_BLOCK], vectorOnes, 12);
  pushTensor("blk.0.attn_q.weight", [K_QUANT_BLOCK, K_QUANT_BLOCK], zeros2x2, 12);
  pushTensor("blk.0.attn_k.weight", [K_QUANT_BLOCK, K_QUANT_BLOCK], zeros2x2, 12);
  pushTensor("blk.0.attn_v.weight", [K_QUANT_BLOCK, K_QUANT_BLOCK], identity2x2, 14);
  pushTensor("blk.0.attn_output.weight", [K_QUANT_BLOCK, K_QUANT_BLOCK], zeros2x2, 12);
  pushTensor("blk.0.ffn_norm.weight", [K_QUANT_BLOCK], vectorOnes, 12);
  pushTensor("blk.0.ffn_gate.weight", [K_QUANT_BLOCK, K_QUANT_BLOCK], zeros2x2, 12);
  pushTensor("blk.0.ffn_down.weight", [K_QUANT_BLOCK, K_QUANT_BLOCK], zeros2x2, 12);
  pushTensor("blk.0.ffn_up.weight", [K_QUANT_BLOCK, K_QUANT_BLOCK], zeros2x2, 12);
  pushTensor("output_norm.weight", [K_QUANT_BLOCK], vectorOnes, 12);
  pushTensor("output.weight", [K_QUANT_BLOCK, 5], outputWeights, 14);

  const directory = Buffer.concat([
    Buffer.from("GGUF"),
    encodeUint32(3),
    encodeUint64(tensors.length),
    encodeUint64(metadata.length),
    ...metadata,
    ...tensors,
  ]);
  const padding = Buffer.alloc((32 - (directory.length % 32)) % 32, 0);

  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, Buffer.concat([directory, padding, ...tensorData]));
}

function writeTiedMixedQ4Q6ZeroLayerGguf(filePath) {
  const metadata = [
    kvString("general.architecture", "llama"),
    kvUint32("general.file_type", 15),
    kvString("tokenizer.ggml.model", "synthetic-tied-q4-q6-k"),
    kvUint32("llama.context_length", 32),
    kvUint32("llama.embedding_length", K_QUANT_BLOCK),
    kvUint32("llama.block_count", 0),
    kvString("tokenizer.chat_template", "{{ prompt }}"),
    kvUint32("tokenizer.ggml.bos_token_id", 0),
    kvUint32("tokenizer.ggml.eos_token_id", 4),
    kvStringArray("tokenizer.ggml.tokens", ["<s>", "hello", " ", "world", "</s>"]),
  ];

  const tokenEmbeddings = encodeQ6KTensor(kQuantRows([
    [0, 0],
    [2, 0],
    [0, 1],
    [1, 0],
    [-1, 0],
  ]));
  const outputNorm = encodeQ4KTensor(kQuantRows([[1, 1]]));

  const tensors = [
    tensorEntry("token_embd.weight", [K_QUANT_BLOCK, 5], 14, 0),
    tensorEntry("output_norm.weight", [K_QUANT_BLOCK], 12, tokenEmbeddings.length),
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
  const tensorData = Buffer.concat([tokenEmbeddings, outputNorm]);

  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, Buffer.concat([directory, padding, tensorData]));
}

function compiledWorkerPath() {
  const candidates = [
    path.resolve(process.cwd(), "native", "lizard-native-runner", "build", "lizard-native-runner.exe"),
    path.resolve(process.cwd(), "native", "lizard-native-runner", "build-cmake", "lizard-native-runner.exe"),
  ];
  return candidates.find((candidate) => fs.existsSync(candidate)) ?? null;
}

test("compiled native C++ worker executes a minimal zero-layer GGUF path", async (t) => {
  const binary = compiledWorkerPath();
  if (!binary) {
    t.skip("compiled native worker binary is not present in build/ or build-cmake/");
    return;
  }

  const root = tempDir();
  const modelPath = path.join(root, "zero-layer.gguf");
  writeZeroLayerGguf(modelPath);
  const gguf = inspectGgufFile(modelPath);

  const client = new NativeWorkerProcessClient({
    launchSpec: {
      command: binary,
      args: [],
    },
  });

  try {
    const handshake = await client.handshake();
    assert.equal(handshake.worker, "lizard-native-runner-cpp");
    assert.equal(
      handshake.cpuBuildProfile,
      "portable-x86-64",
      "compiled release worker must not depend on the build host's AVX/AVX2 instruction set",
    );
    assert.ok(handshake.quantizations.includes("bf16"));

    const loaded = await client.loadModel({
      modelArtifact: {
        modelId: "zero-layer",
        modelPath,
      },
      profile: {
        mmap: true,
        mlock: false,
        gguf,
      },
    });

    assert.equal(loaded.executionReady, true);
    assert.equal(loaded.nativeModelStatus, "minimal-graph-ready");
    assert.equal(loaded.compatibility.executor.status, "supported");
    assert.equal(loaded.executorState.zeroLayerExecutable, true);
    assert.equal(loaded.executorState.tokenizer.ready, true);

    await client.openSession({
      sessionId: "compiled-zero-layer",
      modelId: "zero-layer",
    });

    const prefill = await client.prefill({
      sessionId: "compiled-zero-layer",
      prompt: "hello",
      promptArtifact: {
        renderedPrompt: "hello",
        tokenIds: [1],
        tokenIdMode: "exact-vocab",
      },
      kvPlan: {
        estimatedBytes: 0,
      },
      maxTokens: 2,
    });

    assert.deepEqual(prefill.promptTokenIds, [1]);
    assert.equal(prefill.promptTokenMode, "exact-vocab");
    assert.equal(prefill.nativeModelStatus, "minimal-graph-prefilled");
    assert.deepEqual(prefill.embeddingPreview, [1, 0]);

    const generated = await client.generate({
      sessionId: "compiled-zero-layer",
      maxTokens: 2,
      prefillMs: prefill.prefillMs,
    });

    assert.equal(generated.nativeModelStatus, "minimal-graph-executed");
    assert.deepEqual(generated.generatedTokenIds, [3, 4]);
    assert.equal(generated.output, "world</s>");
    assert.deepEqual(generated.chunks, ["world", "</s>"]);
    assert.equal(generated.compatibility.executor.status, "supported");
  } finally {
    await client.shutdown().catch(() => {});
  }
});

test("compiled native C++ worker reports explicit tokenizer-metadata boundaries for non-executable GGUF files", async (t) => {
  const binary = compiledWorkerPath();
  if (!binary) {
    t.skip("compiled native worker binary is not present in build/ or build-cmake/");
    return;
  }

  const root = tempDir();
  const modelPath = path.join(root, "zero-layer-tokenizerless.gguf");
  writeTokenizerlessZeroLayerGguf(modelPath);
  const gguf = inspectGgufFile(modelPath);

  const client = new NativeWorkerProcessClient({
    launchSpec: {
      command: binary,
      args: [],
    },
  });

  try {
    const loaded = await client.loadModel({
      modelArtifact: {
        modelId: "zero-layer-tokenizerless",
        modelPath,
      },
      profile: {
        mmap: true,
        mlock: false,
        gguf,
      },
    });

    assert.equal(loaded.executionReady, false);
    assert.equal(loaded.nativeModelStatus, "load-only");
    assert.equal(loaded.compatibility.executor.status, "unsupported");
    assert.match(loaded.compatibility.executor.reason, /tokenizer vocabulary metadata required for native prompt and token decoding/i);
    assert.equal(loaded.compatibility.executor.tokenizerMetadataSupported, false);
    assert.equal(loaded.executorState.tokenizer.ready, false);
  } finally {
    await client.shutdown().catch(() => {});
  }
});

test("compiled native C++ worker executes a one-layer llama GGUF path", async (t) => {
  const binary = compiledWorkerPath();
  if (!binary) {
    t.skip("compiled native worker binary is not present in build/ or build-cmake/");
    return;
  }

  const root = tempDir();
  const modelPath = path.join(root, "one-layer.gguf");
  writeOneLayerGguf(modelPath);
  const gguf = inspectGgufFile(modelPath);

  const client = new NativeWorkerProcessClient({
    launchSpec: {
      command: binary,
      args: [],
    },
  });

  try {
    const loaded = await client.loadModel({
      modelArtifact: {
        modelId: "one-layer",
        modelPath,
      },
      profile: {
        mmap: true,
        mlock: false,
        gguf,
      },
    });

    assert.equal(loaded.executionReady, true);
    assert.equal(loaded.nativeModelStatus, "llama-graph-ready");
    assert.equal(loaded.compatibility.executor.status, "supported");
    assert.equal(loaded.executorState.zeroLayerExecutable, false);
    assert.equal(loaded.executorState.graphExecutable, true);

    await client.openSession({
      sessionId: "compiled-one-layer",
      modelId: "one-layer",
    });

    const prefill = await client.prefill({
      sessionId: "compiled-one-layer",
      prompt: "hello",
      promptArtifact: {
        renderedPrompt: "hello",
        tokenIds: [1],
        tokenIdMode: "exact-vocab",
      },
      kvPlan: {
        estimatedBytes: 0,
      },
      maxTokens: 2,
    });

    assert.equal(prefill.nativeModelStatus, "llama-graph-prefilled");
    assert.deepEqual(prefill.promptTokenIds, [1]);
    assert.equal(prefill.cachedTokenCount, 1);
    assert.equal(prefill.sessionMetrics.cachedTokenCount, 1);
    assert.equal(prefill.sessionMetrics.promptTokenCount, 1);
    assert.ok(prefill.sessionMetrics.nativeSessionBytes >= 8);

    const clonedPrefill = await client.cloneSession({
      sourceSessionId: "compiled-one-layer",
      sessionId: "compiled-one-layer-clone",
      maxTokens: 2,
      stopSequences: [],
    });
    assert.equal(clonedPrefill.prefillSnapshotHit, true);
    assert.deepEqual(clonedPrefill.promptTokenIds, prefill.promptTokenIds);
    assert.equal(clonedPrefill.cachedTokenCount, prefill.cachedTokenCount);

    await client.cloneSession({
      sourceSessionId: "compiled-one-layer",
      sessionId: "compiled-one-layer-coalesce-a",
      maxTokens: 2,
      stopSequences: [],
    });
    await client.cloneSession({
      sourceSessionId: "compiled-one-layer",
      sessionId: "compiled-one-layer-coalesce-b",
      maxTokens: 2,
      stopSequences: [],
    });
    const coalescedStep = await client.request("decode_batch", {
      sessionIds: [
        "compiled-one-layer-coalesce-a",
        "compiled-one-layer-coalesce-b",
      ],
    });
    assert.equal(coalescedStep.batchSize, 2);
    assert.equal(coalescedStep.effectiveRows, 1);
    assert.equal(coalescedStep.coalescedSessions, 1);
    assert.deepEqual(
      coalescedStep.tokens.map((entry) => entry.tokenId),
      [3, 3],
    );
    await client.closeSession({ sessionId: "compiled-one-layer-coalesce-a" });
    await client.closeSession({ sessionId: "compiled-one-layer-coalesce-b" });

    const firstDecode = await client.decode({
      sessionId: "compiled-one-layer",
      maxTokens: 1,
      prefillMs: prefill.prefillMs,
    });

    assert.equal(firstDecode.nativeModelStatus, "llama-graph-executed");
    assert.deepEqual(firstDecode.generatedTokenIds, [3]);
    assert.equal(firstDecode.output, "world");
    assert.deepEqual(firstDecode.chunks, ["world"]);
    assert.equal(firstDecode.cachedTokenCount, 2);
    assert.equal(firstDecode.compatibility.executor.status, "supported");

    const cloneFirstDecode = await client.decode({
      sessionId: "compiled-one-layer-clone",
      maxTokens: 1,
      prefillMs: clonedPrefill.prefillMs,
    });
    assert.deepEqual(cloneFirstDecode.generatedTokenIds, firstDecode.generatedTokenIds);
    assert.equal(cloneFirstDecode.output, firstDecode.output);
    assert.equal(cloneFirstDecode.cachedTokenCount, firstDecode.cachedTokenCount);

    const secondDecode = await client.decode({
      sessionId: "compiled-one-layer",
      maxTokens: 1,
      prefillMs: 0,
    });

    assert.equal(secondDecode.nativeModelStatus, "llama-graph-executed");
    assert.deepEqual(secondDecode.generatedTokenIds, []);
    assert.equal(secondDecode.output, "");
    assert.deepEqual(secondDecode.chunks, []);
    assert.equal(secondDecode.cachedTokenCount, 2);
    assert.equal(secondDecode.compatibility.executor.status, "supported");
    assert.equal(secondDecode.sessionMetrics.cachedTokenCount, 2);
    assert.equal(secondDecode.sessionMetrics.generatedTokenCount, 1);
    assert.ok(secondDecode.sessionMetrics.lastDecodeMs >= 1);

    const cloneSecondDecode = await client.decode({
      sessionId: "compiled-one-layer-clone",
      maxTokens: 1,
      prefillMs: 0,
    });
    assert.deepEqual(cloneSecondDecode.generatedTokenIds, secondDecode.generatedTokenIds);
    assert.equal(cloneSecondDecode.output, secondDecode.output);
    assert.equal(cloneSecondDecode.sessionMetrics.generatedTokenCount, 1);

    const metrics = await client.metrics();
    assert.equal(metrics.activeSessionCount, 2);
    assert.equal(metrics.totalKvCacheBytes, 0);
    assert.ok(metrics.totalNativeSessionBytes >= secondDecode.sessionMetrics.nativeSessionBytes);
    assert.equal(metrics.sessions.length, 2);
    for (const sessionMetrics of metrics.sessions) {
      assert.equal(sessionMetrics.cachedTokenCount, 2);
      assert.equal(sessionMetrics.generatedTokenCount, 1);
    }
  } finally {
    await client.shutdown().catch(() => {});
  }
});

test("compiled native C++ worker executes a grouped-query-attention GGUF path", async (t) => {
  const binary = compiledWorkerPath();
  if (!binary) {
    t.skip("compiled native worker binary is not present in build/ or build-cmake/");
    return;
  }

  const root = tempDir();
  const modelPath = path.join(root, "one-layer-gqa.gguf");
  writeGroupedQueryAttentionGguf(modelPath);
  const gguf = inspectGgufFile(modelPath);

  const client = new NativeWorkerProcessClient({
    launchSpec: {
      command: binary,
      args: [],
    },
  });

  try {
    const loaded = await client.loadModel({
      modelArtifact: {
        modelId: "one-layer-gqa",
        modelPath,
      },
      profile: {
        mmap: true,
        mlock: false,
        gguf,
      },
    });

    assert.equal(loaded.executionReady, true);
    assert.equal(loaded.nativeModelStatus, "llama-graph-ready");
    assert.equal(loaded.compatibility.executor.status, "supported");
    assert.equal(loaded.executorState.graphExecutable, true);
    assert.equal(loaded.model.attention?.headCount, 2);
    assert.equal(loaded.model.attention?.kvHeadCount, 1);

    await client.openSession({
      sessionId: "compiled-one-layer-gqa",
      modelId: "one-layer-gqa",
    });

    const prefill = await client.prefill({
      sessionId: "compiled-one-layer-gqa",
      prompt: "hello ",
      promptArtifact: {
        renderedPrompt: "hello ",
        tokenIds: [1, 2],
        tokenIdMode: "exact-vocab",
      },
      kvPlan: {
        estimatedBytes: 0,
      },
      maxTokens: 1,
    });

    assert.equal(prefill.nativeModelStatus, "llama-graph-prefilled");
    assert.deepEqual(prefill.promptTokenIds, [1, 2]);
    assert.equal(prefill.cachedTokenCount, 2);

    const generated = await client.generate({
      sessionId: "compiled-one-layer-gqa",
      maxTokens: 1,
      prefillMs: prefill.prefillMs,
    });

    assert.equal(generated.nativeModelStatus, "llama-graph-executed");
    assert.deepEqual(generated.generatedTokenIds, [3]);
    assert.equal(generated.output, "world");
    assert.deepEqual(generated.chunks, ["world"]);
    assert.equal(generated.cachedTokenCount, 3);
  } finally {
    await client.shutdown().catch(() => {});
  }
});

test("compiled native C++ worker honors seeded sampling settings", async (t) => {
  const binary = compiledWorkerPath();
  if (!binary) {
    t.skip("compiled native worker binary is not present in build/ or build-cmake/");
    return;
  }

  const root = tempDir();
  const modelPath = path.join(root, "sampling-zero-layer.gguf");
  writeSamplingZeroLayerGguf(modelPath);
  const gguf = inspectGgufFile(modelPath);

  const runSeededSession = async (sessionId) => {
    const client = new NativeWorkerProcessClient({
      launchSpec: {
        command: binary,
        args: [],
      },
    });

    try {
      await client.loadModel({
        modelArtifact: {
          modelId: "sampling-zero-layer",
          modelPath,
        },
        profile: {
          mmap: true,
          mlock: false,
          gguf,
        },
      });

      await client.openSession({
        sessionId,
        modelId: "sampling-zero-layer",
      });

      const prefill = await client.prefill({
        sessionId,
        prompt: "hello",
        promptArtifact: {
          renderedPrompt: "hello",
          tokenIds: [1],
          tokenIdMode: "exact-vocab",
        },
        kvPlan: {
          estimatedBytes: 0,
        },
        maxTokens: 1,
      });

      return client.generate({
        sessionId,
        maxTokens: 1,
        prefillMs: prefill.prefillMs,
        sampling: {
          temperature: 1,
          topK: 2,
          topP: 1,
          seed: 42,
        },
      });
    } finally {
      await client.shutdown().catch(() => {});
    }
  };

  const first = await runSeededSession("seeded-a");
  const second = await runSeededSession("seeded-b");

  assert.deepEqual(first.generatedTokenIds, second.generatedTokenIds);
  assert.equal(first.output, second.output);
  assert.ok([3, 4].includes(first.generatedTokenIds[0]));
  assert.equal(first.sampling.seed, 42);
  assert.equal(first.sampling.topK, 2);
  assert.equal(first.sampling.topP, 1);
  assert.equal(first.sampling.temperature, 1);
});

test("compiled native C++ worker executes a q4_k_m one-layer GGUF path", async (t) => {
  const binary = compiledWorkerPath();
  if (!binary) {
    t.skip("compiled native worker binary is not present in build/ or build-cmake/");
    return;
  }

  const root = tempDir();
  const modelPath = path.join(root, "one-layer-q4-k-m.gguf");
  writeQ4KOneLayerGguf(modelPath);
  const gguf = inspectGgufFile(modelPath);

  const client = new NativeWorkerProcessClient({
    launchSpec: {
      command: binary,
      args: [],
    },
  });

  try {
    const loaded = await client.loadModel({
      modelArtifact: {
        modelId: "one-layer-q4-k-m",
        modelPath,
      },
      profile: {
        mmap: true,
        mlock: false,
        gguf,
      },
    });

    assert.equal(loaded.executionReady, true);
    assert.equal(loaded.nativeModelStatus, "llama-graph-ready");
    assert.equal(loaded.model.quantization, "q4_k_m");
    assert.equal(loaded.compatibility.executor.status, "supported");
    assert.equal(loaded.executorState.graphExecutable, true);
    assert.ok(loaded.executorState.supportedDequantTypes.includes("Q4_K"));

    await client.openSession({
      sessionId: "compiled-one-layer-q4-k-m",
      modelId: "one-layer-q4-k-m",
    });

    const prefill = await client.prefill({
      sessionId: "compiled-one-layer-q4-k-m",
      prompt: "hello",
      promptArtifact: {
        renderedPrompt: "hello",
        tokenIds: [1],
        tokenIdMode: "exact-vocab",
      },
      kvPlan: {
        estimatedBytes: 0,
      },
      maxTokens: 2,
    });

    assert.equal(prefill.nativeModelStatus, "llama-graph-prefilled");
    assert.equal(prefill.cachedTokenCount, 1);

    const generated = await client.generate({
      sessionId: "compiled-one-layer-q4-k-m",
      maxTokens: 2,
      prefillMs: prefill.prefillMs,
    });

    assert.equal(generated.nativeModelStatus, "llama-graph-executed");
    assert.deepEqual(generated.generatedTokenIds, [3]);
    assert.equal(generated.output, "world");
    assert.deepEqual(generated.chunks, ["world"]);
    assert.equal(generated.compatibility.executor.status, "supported");
  } finally {
    await client.shutdown().catch(() => {});
  }
});

test("compiled native C++ worker executes a q5_k_m one-layer GGUF path", async (t) => {
  const binary = compiledWorkerPath();
  if (!binary) {
    t.skip("compiled native worker binary is not present in build/ or build-cmake/");
    return;
  }

  const root = tempDir();
  const modelPath = path.join(root, "one-layer-q5-k-m.gguf");
  writeQ5KOneLayerGguf(modelPath);
  const gguf = inspectGgufFile(modelPath);

  const client = new NativeWorkerProcessClient({
    launchSpec: {
      command: binary,
      args: [],
    },
  });

  try {
    const loaded = await client.loadModel({
      modelArtifact: {
        modelId: "one-layer-q5-k-m",
        modelPath,
      },
      profile: {
        mmap: true,
        mlock: false,
        gguf,
      },
    });

    assert.equal(loaded.executionReady, true);
    assert.equal(loaded.nativeModelStatus, "llama-graph-ready");
    assert.equal(loaded.model.quantization, "q5_k_m");
    assert.equal(loaded.compatibility.quantization.key, "q5_k_m");
    assert.equal(loaded.compatibility.executor.status, "supported");
    assert.equal(loaded.executorState.graphExecutable, true);
    assert.ok(loaded.executorState.supportedDequantTypes.includes("Q5_K"));

    await client.openSession({
      sessionId: "compiled-one-layer-q5-k-m",
      modelId: "one-layer-q5-k-m",
    });

    const prefill = await client.prefill({
      sessionId: "compiled-one-layer-q5-k-m",
      prompt: "hello",
      promptArtifact: {
        renderedPrompt: "hello",
        tokenIds: [1],
        tokenIdMode: "exact-vocab",
      },
      kvPlan: {
        estimatedBytes: 0,
      },
      maxTokens: 2,
    });

    assert.equal(prefill.nativeModelStatus, "llama-graph-prefilled");
    assert.equal(prefill.cachedTokenCount, 1);

    const generated = await client.generate({
      sessionId: "compiled-one-layer-q5-k-m",
      maxTokens: 2,
      prefillMs: prefill.prefillMs,
    });

    assert.equal(generated.nativeModelStatus, "llama-graph-executed");
    assert.deepEqual(generated.generatedTokenIds, [3]);
    assert.equal(generated.output, "world");
    assert.deepEqual(generated.chunks, ["world"]);
    assert.equal(generated.compatibility.executor.status, "supported");
  } finally {
    await client.shutdown().catch(() => {});
  }
});

test("compiled native C++ worker executes a mixed q4_k_m and q6_k one-layer GGUF path", async (t) => {
  const binary = compiledWorkerPath();
  if (!binary) {
    t.skip("compiled native worker binary is not present in build/ or build-cmake/");
    return;
  }

  const root = tempDir();
  const modelPath = path.join(root, "one-layer-q4-q6-k.gguf");
  writeMixedQ4Q6OneLayerGguf(modelPath);
  const gguf = inspectGgufFile(modelPath);

  const client = new NativeWorkerProcessClient({
    launchSpec: {
      command: binary,
      args: [],
    },
  });

  try {
    const loaded = await client.loadModel({
      modelArtifact: {
        modelId: "one-layer-q4-q6-k",
        modelPath,
      },
      profile: {
        mmap: true,
        mlock: false,
        gguf,
      },
    });

    assert.equal(loaded.executionReady, true);
    assert.equal(loaded.nativeModelStatus, "llama-graph-ready");
    assert.equal(loaded.model.quantization, "q4_k_m");
    assert.equal(loaded.compatibility.executor.status, "supported");
    assert.equal(loaded.executorState.graphExecutable, true);
    assert.ok(loaded.executorState.supportedDequantTypes.includes("Q4_K"));
    assert.ok(loaded.executorState.supportedDequantTypes.includes("Q6_K"));

    await client.openSession({
      sessionId: "compiled-one-layer-q4-q6-k",
      modelId: "one-layer-q4-q6-k",
    });

    const prefill = await client.prefill({
      sessionId: "compiled-one-layer-q4-q6-k",
      prompt: "hello",
      promptArtifact: {
        renderedPrompt: "hello",
        tokenIds: [1],
        tokenIdMode: "exact-vocab",
      },
      kvPlan: {
        estimatedBytes: 0,
      },
      maxTokens: 2,
    });

    assert.equal(prefill.nativeModelStatus, "llama-graph-prefilled");
    assert.equal(prefill.cachedTokenCount, 1);

    const generated = await client.generate({
      sessionId: "compiled-one-layer-q4-q6-k",
      maxTokens: 2,
      prefillMs: prefill.prefillMs,
    });

    assert.equal(generated.nativeModelStatus, "llama-graph-executed");
    assert.deepEqual(generated.generatedTokenIds, [3]);
    assert.equal(generated.output, "world");
    assert.deepEqual(generated.chunks, ["world"]);
    assert.equal(generated.compatibility.executor.status, "supported");
  } finally {
    await client.shutdown().catch(() => {});
  }
});

test("compiled native C++ worker executes a tied-output GGUF path without output.weight", async (t) => {
  const binary = compiledWorkerPath();
  if (!binary) {
    t.skip("compiled native worker binary is not present in build/ or build-cmake/");
    return;
  }

  const root = tempDir();
  const modelPath = path.join(root, "zero-layer-tied-q4-q6-k.gguf");
  writeTiedMixedQ4Q6ZeroLayerGguf(modelPath);
  const gguf = inspectGgufFile(modelPath);

  const client = new NativeWorkerProcessClient({
    launchSpec: {
      command: binary,
      args: [],
    },
  });

  try {
    const loaded = await client.loadModel({
      modelArtifact: {
        modelId: "zero-layer-tied-q4-q6-k",
        modelPath,
      },
      profile: {
        mmap: true,
        mlock: false,
        gguf,
      },
    });

    assert.equal(loaded.executionReady, true);
    assert.equal(loaded.nativeModelStatus, "minimal-graph-ready");
    assert.equal(loaded.compatibility.executor.status, "supported");
    assert.equal(loaded.executorState.zeroLayerExecutable, true);
    assert.equal(loaded.executorState.outputTied, true);
    assert.deepEqual(loaded.executorState.missingShared, []);
    assert.ok(loaded.executorState.supportedDequantTypes.includes("Q4_K"));
    assert.ok(loaded.executorState.supportedDequantTypes.includes("Q6_K"));

    await client.openSession({
      sessionId: "compiled-zero-layer-tied-q4-q6-k",
      modelId: "zero-layer-tied-q4-q6-k",
    });

    const prefill = await client.prefill({
      sessionId: "compiled-zero-layer-tied-q4-q6-k",
      prompt: "hello",
      promptArtifact: {
        renderedPrompt: "hello",
        tokenIds: [1],
        tokenIdMode: "exact-vocab",
      },
      kvPlan: {
        estimatedBytes: 0,
      },
      maxTokens: 2,
    });

    const generated = await client.generate({
      sessionId: "compiled-zero-layer-tied-q4-q6-k",
      maxTokens: 2,
      prefillMs: prefill.prefillMs,
    });

    assert.equal(generated.nativeModelStatus, "minimal-graph-executed");
    assert.deepEqual(generated.generatedTokenIds, [1, 1]);
    assert.equal(generated.output, "hellohello");
    assert.deepEqual(generated.chunks, ["hello", "hello"]);
  } finally {
    await client.shutdown().catch(() => {});
  }
});

test("compiled native C++ worker prepends BOS when it infers prompt token IDs internally", async (t) => {
  const binary = compiledWorkerPath();
  if (!binary) {
    t.skip("compiled native worker binary is not present in build/ or build-cmake/");
    return;
  }

  const root = tempDir();
  const modelPath = path.join(root, "zero-layer-bos-inferred.gguf");
  writeZeroLayerGguf(modelPath);
  const gguf = inspectGgufFile(modelPath);

  const client = new NativeWorkerProcessClient({
    launchSpec: {
      command: binary,
      args: [],
    },
  });

  try {
    await client.loadModel({
      modelArtifact: {
        modelId: "zero-layer-bos-inferred",
        modelPath,
      },
      profile: {
        mmap: true,
        mlock: false,
        gguf,
      },
    });

    await client.openSession({
      sessionId: "compiled-zero-layer-bos-inferred",
      modelId: "zero-layer-bos-inferred",
    });

    const prefill = await client.prefill({
      sessionId: "compiled-zero-layer-bos-inferred",
      prompt: "hello",
      promptArtifact: {
        renderedPrompt: "hello",
      },
      kvPlan: {
        estimatedBytes: 0,
      },
      maxTokens: 1,
    });

    assert.deepEqual(prefill.promptTokenIds, [0, 1]);
    assert.equal(prefill.promptTokenMode, "exact-vocab");
  } finally {
    await client.shutdown().catch(() => {});
  }
});

test("compiled native C++ worker infers GPT-2 BPE prompt token IDs internally", async (t) => {
  const binary = compiledWorkerPath();
  if (!binary) {
    t.skip("compiled native worker binary is not present in build/ or build-cmake/");
    return;
  }

  const root = tempDir();
  const modelPath = path.join(root, "zero-layer-gpt2-bpe.gguf");
  writeGpt2BpeZeroLayerGguf(modelPath);
  const gguf = inspectGgufFile(modelPath);

  const client = new NativeWorkerProcessClient({
    launchSpec: {
      command: binary,
      args: [],
    },
  });

  try {
    const loaded = await client.loadModel({
      modelArtifact: {
        modelId: "zero-layer-gpt2-bpe",
        modelPath,
      },
      profile: {
        mmap: true,
        mlock: false,
        gguf,
      },
    });

    assert.equal(loaded.executorState.tokenizer.gpt2PromptTokenizationAvailable, true);

    await client.openSession({
      sessionId: "compiled-zero-layer-gpt2-bpe",
      modelId: "zero-layer-gpt2-bpe",
    });

    const prefill = await client.prefill({
      sessionId: "compiled-zero-layer-gpt2-bpe",
      prompt: " Hello",
      promptArtifact: {
        renderedPrompt: " Hello",
      },
      kvPlan: {
        estimatedBytes: 0,
      },
      maxTokens: 1,
    });

    assert.deepEqual(prefill.promptTokenIds, [0, 9]);
    assert.equal(prefill.promptTokenMode, "gpt2-bpe");
    assert.equal(prefill.nativeModelStatus, "minimal-graph-prefilled");
  } finally {
    await client.shutdown().catch(() => {});
  }
});

test("compiled native C++ worker infers UTF-8 GPT-2 BPE prompt token IDs internally", async (t) => {
  const binary = compiledWorkerPath();
  if (!binary) {
    t.skip("compiled native worker binary is not present in build/ or build-cmake/");
    return;
  }

  const root = tempDir();
  const modelPath = path.join(root, "zero-layer-gpt2-bpe-utf8.gguf");
  writeUtf8Gpt2BpeZeroLayerGguf(modelPath);
  const gguf = inspectGgufFile(modelPath);

  const client = new NativeWorkerProcessClient({
    launchSpec: {
      command: binary,
      args: [],
    },
  });

  try {
    const loaded = await client.loadModel({
      modelArtifact: {
        modelId: "zero-layer-gpt2-bpe-utf8",
        modelPath,
      },
      profile: {
        mmap: true,
        mlock: false,
        gguf,
      },
    });

    assert.equal(loaded.executorState.tokenizer.gpt2PromptTokenizationAvailable, true);

    await client.openSession({
      sessionId: "compiled-zero-layer-gpt2-bpe-utf8",
      modelId: "zero-layer-gpt2-bpe-utf8",
    });

    const prefill = await client.prefill({
      sessionId: "compiled-zero-layer-gpt2-bpe-utf8",
      prompt: " é",
      promptArtifact: {
        renderedPrompt: " é",
      },
      kvPlan: {
        estimatedBytes: 0,
      },
      maxTokens: 1,
    });

    assert.deepEqual(prefill.promptTokenIds, [0, 4]);
    assert.equal(prefill.promptTokenMode, "gpt2-bpe");
    assert.equal(prefill.nativeModelStatus, "minimal-graph-prefilled");
  } finally {
    await client.shutdown().catch(() => {});
  }
});

test("compiled native C++ worker executes a BF16 zero-layer GGUF path", async (t) => {
  const binary = compiledWorkerPath();
  if (!binary) {
    t.skip("compiled native worker binary is not present in build/ or build-cmake/");
    return;
  }

  const root = tempDir();
  const modelPath = path.join(root, "zero-layer-bf16.gguf");
  writeBfloat16ZeroLayerGguf(modelPath);
  const gguf = inspectGgufFile(modelPath);

  const client = new NativeWorkerProcessClient({
    launchSpec: {
      command: binary,
      args: [],
    },
  });

  try {
    const loaded = await client.loadModel({
      modelArtifact: {
        modelId: "zero-layer-bf16",
        modelPath,
      },
      profile: {
        mmap: true,
        mlock: false,
        gguf,
      },
    });

    assert.equal(loaded.executionReady, true);
    assert.equal(loaded.compatibility.quantization.key, "bf16");
    assert.equal(loaded.compatibility.quantization.status, "supported");
    assert.ok(loaded.executorState.supportedDequantTypes.includes("BF16"));

    await client.openSession({
      sessionId: "compiled-zero-layer-bf16",
      modelId: "zero-layer-bf16",
    });

    const prefill = await client.prefill({
      sessionId: "compiled-zero-layer-bf16",
      prompt: "hello",
      promptArtifact: {
        renderedPrompt: "hello",
        tokenIds: [1],
        tokenIdMode: "exact-vocab",
      },
      kvPlan: {
        estimatedBytes: 0,
      },
      maxTokens: 1,
    });

    assert.equal(prefill.nativeModelStatus, "minimal-graph-prefilled");
    assert.deepEqual(prefill.promptTokenIds, [1]);

    const generated = await client.generate({
      sessionId: "compiled-zero-layer-bf16",
      maxTokens: 1,
      prefillMs: prefill.prefillMs,
    });

    assert.equal(generated.nativeModelStatus, "minimal-graph-executed");
    assert.deepEqual(generated.generatedTokenIds, [3]);
    assert.equal(generated.output, "world");
  } finally {
    await client.shutdown().catch(() => {});
  }
});
