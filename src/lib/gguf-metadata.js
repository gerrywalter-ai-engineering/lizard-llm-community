import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

const GGUF_MAGIC = "GGUF";
const DEFAULT_HEADER_READ_BYTES = 4 * 1024 * 1024;
const MAX_HEADER_READ_BYTES = 128 * 1024 * 1024;

const VALUE_TYPES = {
  UINT8: 0,
  INT8: 1,
  UINT16: 2,
  INT16: 3,
  UINT32: 4,
  INT32: 5,
  FLOAT32: 6,
  BOOL: 7,
  STRING: 8,
  ARRAY: 9,
  UINT64: 10,
  INT64: 11,
  FLOAT64: 12,
};

const FILE_TYPE_TO_QUANT = {
  0: "F32",
  1: "F16",
  2: "Q4_0",
  3: "Q4_1",
  7: "Q8_0",
  8: "Q5_0",
  9: "Q5_1",
  10: "Q2_K",
  11: "Q3_K_S",
  12: "Q3_K_M",
  13: "Q3_K_L",
  14: "Q4_K_S",
  15: "Q4_K_M",
  16: "Q5_K_S",
  17: "Q5_K_M",
  18: "Q6_K",
  19: "IQ2_XXS",
  20: "IQ2_XS",
  21: "Q2_K_S",
  22: "IQ3_XS",
  23: "IQ3_XXS",
  24: "IQ1_S",
  25: "IQ4_NL",
  26: "IQ3_S",
  27: "IQ3_M",
  28: "IQ2_S",
  29: "IQ2_M",
  30: "IQ4_XS",
  31: "IQ1_M",
  32: "BF16",
  36: "TQ1_0",
  37: "TQ2_0",
};

const GGML_TYPE_LAYOUTS = {
  0: { name: "F32", blockSize: 1, bytesPerBlock: 4 },
  1: { name: "F16", blockSize: 1, bytesPerBlock: 2 },
  30: { name: "BF16", blockSize: 1, bytesPerBlock: 2 },
  2: { name: "Q4_0", blockSize: 32, bytesPerBlock: 18 },
  3: { name: "Q4_1", blockSize: 32, bytesPerBlock: 20 },
  6: { name: "Q5_0", blockSize: 32, bytesPerBlock: 22 },
  7: { name: "Q5_1", blockSize: 32, bytesPerBlock: 24 },
  8: { name: "Q8_0", blockSize: 32, bytesPerBlock: 34 },
  9: { name: "Q8_1", blockSize: 32, bytesPerBlock: 36 },
  10: { name: "Q2_K", blockSize: 256, bytesPerBlock: 84 },
  11: { name: "Q3_K", blockSize: 256, bytesPerBlock: 110 },
  12: { name: "Q4_K", blockSize: 256, bytesPerBlock: 144 },
  13: { name: "Q5_K", blockSize: 256, bytesPerBlock: 176 },
  14: { name: "Q6_K", blockSize: 256, bytesPerBlock: 210 },
  15: { name: "Q8_K", blockSize: 256, bytesPerBlock: 292 },
  20: { name: "IQ4_NL", blockSize: 32, bytesPerBlock: 18 },
  23: { name: "IQ4_XS", blockSize: 256, bytesPerBlock: 136 },
};

function hashJson(value) {
  return crypto.createHash("sha256").update(JSON.stringify(value ?? null)).digest("hex");
}

function readHeaderBuffer(filePath, bytes = DEFAULT_HEADER_READ_BYTES) {
  const stat = fs.statSync(filePath);
  const bufferSize = Math.min(stat.size, bytes);
  const fd = fs.openSync(filePath, "r");
  try {
    const buffer = Buffer.alloc(bufferSize);
    fs.readSync(fd, buffer, 0, bufferSize, 0);
    return { buffer, size: stat.size };
  } finally {
    fs.closeSync(fd);
  }
}

class BufferReader {
  constructor(buffer) {
    this.buffer = buffer;
    this.offset = 0;
  }

  remaining() {
    return this.buffer.length - this.offset;
  }

  ensure(size) {
    if (this.offset + size > this.buffer.length) {
      throw new Error("GGUF metadata exceeds buffered header window.");
    }
  }

  readUInt8() {
    this.ensure(1);
    const value = this.buffer.readUInt8(this.offset);
    this.offset += 1;
    return value;
  }

  readInt8() {
    this.ensure(1);
    const value = this.buffer.readInt8(this.offset);
    this.offset += 1;
    return value;
  }

  readUInt16() {
    this.ensure(2);
    const value = this.buffer.readUInt16LE(this.offset);
    this.offset += 2;
    return value;
  }

  readInt16() {
    this.ensure(2);
    const value = this.buffer.readInt16LE(this.offset);
    this.offset += 2;
    return value;
  }

  readUInt32() {
    this.ensure(4);
    const value = this.buffer.readUInt32LE(this.offset);
    this.offset += 4;
    return value;
  }

  readInt32() {
    this.ensure(4);
    const value = this.buffer.readInt32LE(this.offset);
    this.offset += 4;
    return value;
  }

  readFloat32() {
    this.ensure(4);
    const value = this.buffer.readFloatLE(this.offset);
    this.offset += 4;
    return value;
  }

  readFloat64() {
    this.ensure(8);
    const value = this.buffer.readDoubleLE(this.offset);
    this.offset += 8;
    return value;
  }

  readBool() {
    return this.readUInt8() !== 0;
  }

  readUInt64() {
    this.ensure(8);
    const value = this.buffer.readBigUInt64LE(this.offset);
    this.offset += 8;
    if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
      return Number.MAX_SAFE_INTEGER;
    }
    return Number(value);
  }

  readInt64() {
    this.ensure(8);
    const value = this.buffer.readBigInt64LE(this.offset);
    this.offset += 8;
    if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
      return Number.MAX_SAFE_INTEGER;
    }
    if (value < BigInt(Number.MIN_SAFE_INTEGER)) {
      return Number.MIN_SAFE_INTEGER;
    }
    return Number(value);
  }

  readString() {
    const length = this.readUInt64();
    this.ensure(length);
    const value = this.buffer.toString("utf8", this.offset, this.offset + length);
    this.offset += length;
    return value;
  }

  readValue(type) {
    switch (type) {
      case VALUE_TYPES.UINT8:
        return this.readUInt8();
      case VALUE_TYPES.INT8:
        return this.readInt8();
      case VALUE_TYPES.UINT16:
        return this.readUInt16();
      case VALUE_TYPES.INT16:
        return this.readInt16();
      case VALUE_TYPES.UINT32:
        return this.readUInt32();
      case VALUE_TYPES.INT32:
        return this.readInt32();
      case VALUE_TYPES.FLOAT32:
        return this.readFloat32();
      case VALUE_TYPES.BOOL:
        return this.readBool();
      case VALUE_TYPES.STRING:
        return this.readString();
      case VALUE_TYPES.ARRAY: {
        const elementType = this.readUInt32();
        const length = this.readUInt64();
        const values = [];
        for (let index = 0; index < length; index += 1) {
          values.push(this.readValue(elementType));
        }
        return values;
      }
      case VALUE_TYPES.UINT64:
        return this.readUInt64();
      case VALUE_TYPES.INT64:
        return this.readInt64();
      case VALUE_TYPES.FLOAT64:
        return this.readFloat64();
      default:
        throw new Error(`Unsupported GGUF metadata value type: ${type}`);
    }
  }
}

function lookupFirst(metadata, keys = [], suffixes = []) {
  for (const key of keys) {
    if (metadata[key] != null) {
      return metadata[key];
    }
  }
  for (const [key, value] of Object.entries(metadata)) {
    if (suffixes.some((suffix) => key.endsWith(suffix)) && value != null) {
      return value;
    }
  }
  return null;
}

// Cross-layer value sharing detection. Some architectures (Gemma 4 is the
// first observed) legitimately omit blk.N.attn_v.weight on layers that reuse
// a neighbor's value projection. The native executor requires a value tensor
// on every layer, so a compact summary of the gaps lets the runtime-provider
// pre-flight route such models to llama.cpp automatically instead of letting
// the native worker refuse the load mid-benchmark.
export function summarizeAttentionValueGaps(tensorNames = [], blockCount = null) {
  const totalLayers = Number(blockCount ?? 0);
  if (!Number.isFinite(totalLayers) || totalLayers <= 0) {
    return null;
  }
  const names = tensorNames instanceof Set ? tensorNames : new Set(tensorNames);
  const missingLayers = [];
  let presentLayerCount = 0;
  for (let layer = 0; layer < totalLayers; layer += 1) {
    const prefix = `blk.${layer}.`;
    if (names.has(`${prefix}attn_v.weight`) || names.has(`${prefix}attn_qkv.weight`)) {
      presentLayerCount += 1;
    } else {
      missingLayers.push(layer);
    }
  }
  if (missingLayers.length === 0 || presentLayerCount === 0) {
    // A complete directory has no gaps to report; a layout where NO layer
    // carries a value projection is a different architecture family, owned
    // by the architecture matrix rather than this summary.
    return null;
  }
  return {
    missingLayers: missingLayers.slice(0, 16),
    missingLayerCount: missingLayers.length,
    presentLayerCount,
    totalLayers,
  };
}

function buildTensorPreview(tensors = [], limit = 8) {
  return tensors.slice(0, limit).map((tensor) => ({
    name: tensor.name,
    dimensions: tensor.dimensions,
    ggmlType: tensor.ggmlType,
    ggmlTypeName: tensor.ggmlTypeName ?? null,
    elementCount: tensor.elementCount ?? null,
    byteLength: tensor.byteLength ?? null,
    offsetBytes: tensor.offsetBytes,
    absoluteOffsetBytes: tensor.absoluteOffsetBytes ?? null,
  }));
}

function alignOffset(offset, alignment = 32) {
  const normalized = Number.isFinite(alignment) && alignment > 0 ? alignment : 32;
  const remainder = offset % normalized;
  return remainder === 0 ? offset : offset + normalized - remainder;
}

function tensorElementCount(dimensions = []) {
  return dimensions.reduce((total, dimension) => total * Number(dimension || 0), 1);
}

function tensorByteLength({ ggmlType, dimensions = [] }) {
  const layout = GGML_TYPE_LAYOUTS[ggmlType] ?? null;
  if (!layout) {
    return {
      ggmlTypeName: null,
      elementCount: tensorElementCount(dimensions),
      byteLength: null,
    };
  }
  const elementCount = tensorElementCount(dimensions);
  const blockCount = Math.ceil(elementCount / layout.blockSize);
  return {
    ggmlTypeName: layout.name,
    elementCount,
    byteLength: blockCount * layout.bytesPerBlock,
  };
}

function findTensorByNames(tensorMap, names = []) {
  for (const name of names) {
    if (tensorMap.has(name)) {
      return tensorMap.get(name);
    }
  }
  return null;
}

function analyzeLlamaTensorLayout({ tensors = [], blockCount = null }) {
  const tensorMap = new Map(tensors.map((tensor) => [tensor.name, tensor]));
  const tensorTypes = [...new Set(
    tensors.map((tensor) => tensor.ggmlTypeName ?? `TYPE_${tensor.ggmlType}`),
  )];
  const shared = {
    tokenEmbeddings: Boolean(findTensorByNames(tensorMap, ["token_embd.weight", "tok_embeddings.weight"])),
    outputNorm: Boolean(findTensorByNames(tensorMap, ["output_norm.weight", "norm.weight"])),
    output: Boolean(findTensorByNames(tensorMap, ["output.weight", "lm_head.weight"])),
  };
  const totalLayers = Number.isFinite(blockCount) && blockCount > 0 ? blockCount : 0;
  let completeLayers = 0;
  const incompleteLayers = [];

  for (let layer = 0; layer < totalLayers; layer += 1) {
    const prefix = `blk.${layer}.`;
    const required = {
      attentionNorm: Boolean(findTensorByNames(tensorMap, [`${prefix}attn_norm.weight`])),
      query: Boolean(findTensorByNames(tensorMap, [`${prefix}attn_q.weight`])),
      key: Boolean(findTensorByNames(tensorMap, [`${prefix}attn_k.weight`])),
      value: Boolean(findTensorByNames(tensorMap, [`${prefix}attn_v.weight`])),
      output: Boolean(findTensorByNames(tensorMap, [`${prefix}attn_output.weight`])),
      ffnNorm: Boolean(findTensorByNames(tensorMap, [`${prefix}ffn_norm.weight`])),
      gate: Boolean(findTensorByNames(tensorMap, [`${prefix}ffn_gate.weight`])),
      down: Boolean(findTensorByNames(tensorMap, [`${prefix}ffn_down.weight`])),
      up: Boolean(findTensorByNames(tensorMap, [`${prefix}ffn_up.weight`])),
    };
    const missing = Object.entries(required)
      .filter(([, present]) => !present)
      .map(([name]) => name);
    if (missing.length === 0) {
      completeLayers += 1;
      continue;
    }
    incompleteLayers.push({
      layer,
      missing,
    });
  }

  return {
    shared,
    tensorTypes,
    totalLayers,
    completeLayers,
    incompleteLayers,
    tensorDirectoryComplete: shared.tokenEmbeddings && shared.outputNorm && shared.output && (
      totalLayers === 0 || completeLayers === totalLayers
    ),
  };
}

function normalizeQuant(value, fallbackQuant = null, filePath = null) {
  if (typeof value === "number" && FILE_TYPE_TO_QUANT[value]) {
    return FILE_TYPE_TO_QUANT[value];
  }
  if (typeof value === "string" && value.trim()) {
    return value.trim().toUpperCase();
  }
  const filename = path.basename(filePath ?? "");
  const match = filename.match(/(Q\d(?:_[A-Z0-9]+)?|IQ\d(?:_[A-Z0-9]+)?|TQ\d(?:_[A-Z0-9]+)?|BF16|F16|F32)/i);
  return match ? match[1].toUpperCase() : fallbackQuant;
}

function inferParameterEstimate({ embeddingLength, blockCount, fileSizeBytes, quant }) {
  if (!embeddingLength || !blockCount || !fileSizeBytes) {
    return null;
  }
  const bitsPerWeightGuess = (() => {
    switch (normalizeQuant(quant)) {
      case "Q2_K":
      case "Q2_K_S":
        return 2.6;
      case "TQ1_0":
        return 1.69;
      case "IQ1_S":
        return 1.6;
      case "IQ1_M":
        return 1.75;
      case "IQ2_XXS":
        return 2.1;
      case "IQ2_XS":
        return 2.3;
      case "IQ2_S":
        return 2.5;
      case "IQ2_M":
        return 2.7;
      case "Q3_K_M":
      case "Q3_K_S":
      case "Q3_K_L":
        return 3.4;
      case "IQ3_XXS":
        return 3.1;
      case "IQ3_XS":
        return 3.3;
      case "IQ3_S":
        return 3.4;
      case "IQ3_M":
        return 3.6;
      case "Q4_K_M":
      case "Q4_K_S":
      case "Q4_0":
      case "Q4_1":
      case "IQ4_XS":
      case "IQ4_NL":
        return 4.5;
      case "Q5_K_M":
      case "Q5_K_S":
      case "Q5_0":
      case "Q5_1":
        return 5.5;
      case "Q6_K":
        return 6.4;
      case "Q8_0":
        return 8;
      case "F16":
      case "BF16":
        return 16;
      case "F32":
        return 32;
      default:
        return 4.5;
    }
  })();
  const paramsEstimate = (fileSizeBytes * 8) / bitsPerWeightGuess / 1_000_000_000;
  return Number(paramsEstimate.toFixed(2));
}

export function inspectGgufFile(filePath, {
  fallbackQuant = null,
  fallbackArchitecture = null,
  fallbackParamsB = null,
} = {}) {
  const fileSizeBytes = fs.statSync(filePath).size;
  const result = {
    metadataVersion: 1,
    status: "incomplete",
    warnings: [],
    fileSizeBytes,
    fileSizeGb: Number((fileSizeBytes / 1024 / 1024 / 1024).toFixed(3)),
    metadataHash: null,
    architecture: fallbackArchitecture,
    quant: fallbackQuant,
    tokenizer: null,
    contextLength: null,
    embeddingLength: null,
    blockCount: null,
    expertCount: null,
    expertUsedCount: null,
    tensorCount: null,
    parameterEstimateB: fallbackParamsB ?? null,
    specialTokens: {},
    specialTokenTexts: {},
    chatTemplate: null,
    tensorDirectoryLoaded: false,
    tensorDataOffsetBytes: null,
    tensorDataBytes: null,
    tensorsPreview: [],
    attentionValueGaps: null,
    tokenVocabularySize: null,
    tokenPreview: [],
    tokenMergeCount: null,
    mergePreview: [],
    executorShape: null,
    raw: {},
  };

  let bufferBytes = Math.min(fileSizeBytes, DEFAULT_HEADER_READ_BYTES);
  let parseError = null;

  while (bufferBytes <= Math.min(fileSizeBytes, MAX_HEADER_READ_BYTES)) {
    const { buffer } = readHeaderBuffer(filePath, bufferBytes);
    if (buffer.length < 4 || buffer.toString("utf8", 0, 4) !== GGUF_MAGIC) {
      const error = new Error(`File does not start with the ${GGUF_MAGIC} magic header.`);
      error.code = "NOT_GGUF";
      throw error;
    }
    if (buffer.length < 24) {
      result.warnings.push("GGUF header is too short to parse metadata.");
      result.metadataHash = hashJson({
        fileSizeBytes: result.fileSizeBytes,
        quant: result.quant,
        architecture: result.architecture,
      });
      return result;
    }

    try {
      const reader = new BufferReader(buffer);
      reader.offset = 4;
      const version = reader.readUInt32();
      const tensorCount = reader.readUInt64();
      const metadataKvCount = reader.readUInt64();
      const raw = {};

      for (let index = 0; index < metadataKvCount; index += 1) {
        const key = reader.readString();
        const type = reader.readUInt32();
        raw[key] = reader.readValue(type);
      }
      const tensors = [];
      for (let index = 0; index < tensorCount; index += 1) {
        const name = reader.readString();
        const dimensionCount = reader.readUInt32();
        const dimensions = [];
        for (let dimensionIndex = 0; dimensionIndex < dimensionCount; dimensionIndex += 1) {
          dimensions.push(reader.readUInt64());
        }
        const ggmlType = reader.readUInt32();
        const offsetBytes = reader.readUInt64();
        tensors.push({
          name,
          dimensions,
          ggmlType,
          offsetBytes: String(offsetBytes),
        });
      }
      const alignment = lookupFirst(raw, ["general.alignment"], [".alignment"]);
      const tensorDataOffsetBytes = alignOffset(reader.offset, Number.isFinite(alignment) ? alignment : 32);
      let tensorDataBytes = 0;
      for (const tensor of tensors) {
        const layout = tensorByteLength(tensor);
        tensor.ggmlTypeName = layout.ggmlTypeName;
        tensor.elementCount = layout.elementCount;
        tensor.byteLength = layout.byteLength;
        tensor.absoluteOffsetBytes = String(tensorDataOffsetBytes + Number(tensor.offsetBytes));
        if (layout.byteLength != null) {
          tensorDataBytes = Math.max(tensorDataBytes, Number(tensor.offsetBytes) + layout.byteLength);
        }
      }

      const architecture = lookupFirst(raw, ["general.architecture"], [".architecture"]) ?? fallbackArchitecture;
      const quant = normalizeQuant(
        lookupFirst(raw, ["general.file_type", "general.quantization_version"], [".file_type"]),
        fallbackQuant,
        filePath,
      );
      const tokenizer = lookupFirst(raw, ["tokenizer.ggml.model", "tokenizer.model"], [".tokenizer_model"]);
      const contextLength = lookupFirst(raw, ["llama.context_length", "general.context_length"], [".context_length"]);
      const embeddingLength = lookupFirst(raw, ["llama.embedding_length", "general.embedding_length"], [".embedding_length"]);
      const blockCount = lookupFirst(raw, ["llama.block_count", "general.block_count"], [".block_count"]);
      // expert_count distinguishes Mixtral-style MoE GGUFs, whose declared
      // architecture stays `llama`; the family resolver needs it.
      const expertCount = lookupFirst(raw, [], [".expert_count"]);
      // expert_used_count = how many experts run per token. For an MoE model
      // active params << total params, which is what actually determines
      // runtime memory-bandwidth cost (spec s12).
      const expertUsedCount = lookupFirst(raw, [], [".expert_used_count"]);
      const chatTemplate = lookupFirst(raw, ["tokenizer.chat_template", "general.chat_template"], [".chat_template"]);
      const specialTokens = {
        bos: lookupFirst(raw, ["tokenizer.ggml.bos_token_id"], [".bos_token_id"]),
        eos: lookupFirst(raw, ["tokenizer.ggml.eos_token_id"], [".eos_token_id"]),
        pad: lookupFirst(raw, ["tokenizer.ggml.padding_token_id"], [".padding_token_id"]),
        unk: lookupFirst(raw, ["tokenizer.ggml.unknown_token_id"], [".unknown_token_id"]),
      };
      const tokenVocabulary = Array.isArray(raw["tokenizer.ggml.tokens"])
        ? raw["tokenizer.ggml.tokens"].filter((value) => typeof value === "string")
        : [];
      const tokenMerges = Array.isArray(raw["tokenizer.ggml.merges"])
        ? raw["tokenizer.ggml.merges"].filter((value) => typeof value === "string")
        : [];
      const specialTokenTexts = Object.fromEntries(
        Object.entries(specialTokens)
          .map(([key, index]) => [key, Number.isInteger(index) ? tokenVocabulary[index] ?? null : null])
          .filter(([, value]) => value != null),
      );
      const executorShape = architecture === "llama"
        ? analyzeLlamaTensorLayout({
          tensors,
          blockCount: Number.isFinite(blockCount) ? blockCount : null,
        })
        : null;

      const metadataSummary = {
        version,
        tensorCount,
        metadataKvCount,
        architecture,
        quant,
        tokenizer,
        contextLength,
        embeddingLength,
        blockCount,
        chatTemplate,
        specialTokens,
      };

      result.status = Object.keys(raw).length > 0 ? "ready" : "incomplete";
      result.metadataHash = hashJson(raw);
      result.architecture = architecture;
      result.quant = quant;
      result.tokenizer = tokenizer;
      result.contextLength = Number.isFinite(contextLength) ? contextLength : null;
      result.embeddingLength = Number.isFinite(embeddingLength) ? embeddingLength : null;
      result.blockCount = Number.isFinite(blockCount) ? blockCount : null;
      result.expertCount = Number.isFinite(expertCount) ? expertCount : null;
      result.expertUsedCount = Number.isFinite(expertUsedCount) ? expertUsedCount : null;
      result.tensorCount = tensorCount;
      result.parameterEstimateB = fallbackParamsB ?? inferParameterEstimate({
        embeddingLength: result.embeddingLength,
        blockCount: result.blockCount,
        fileSizeBytes,
        quant,
      });
      result.specialTokens = Object.fromEntries(
        Object.entries(specialTokens).filter(([, value]) => value != null),
      );
      result.specialTokenTexts = specialTokenTexts;
      result.chatTemplate = typeof chatTemplate === "string" ? chatTemplate : null;
      result.tensorDirectoryLoaded = true;
      result.tensorDataOffsetBytes = tensorDataOffsetBytes;
      result.tensorDataBytes = tensorDataBytes || null;
      result.tensorsPreview = buildTensorPreview(tensors);
      result.attentionValueGaps = summarizeAttentionValueGaps(
        tensors.map((tensor) => tensor.name),
        result.blockCount,
      );
      result.tokenVocabularySize = tokenVocabulary.length || null;
      result.tokenPreview = tokenVocabulary.slice(0, 8);
      result.tokenMergeCount = tokenMerges.length || null;
      result.mergePreview = tokenMerges.slice(0, 8);
      result.executorShape = executorShape;
      result.raw = raw;

      if (!result.architecture) {
        result.warnings.push("GGUF architecture metadata is missing.");
      }
      if (!result.contextLength) {
        result.warnings.push("GGUF context length metadata is missing.");
      }
      if (!result.quant) {
        result.warnings.push("GGUF quantization metadata is missing.");
      }
      if (result.warnings.length === 0) {
        result.status = "ready";
      } else if (Object.keys(raw).length > 0) {
        result.status = "partial";
      }

      result.summary = metadataSummary;
      return result;
    } catch (error) {
      parseError = error;
      const headerWindowExceeded = String(error?.message ?? "").includes("GGUF metadata exceeds buffered header window.");
      if (!headerWindowExceeded || bufferBytes >= fileSizeBytes || bufferBytes >= MAX_HEADER_READ_BYTES) {
        break;
      }
      bufferBytes = Math.min(fileSizeBytes, MAX_HEADER_READ_BYTES, bufferBytes * 2);
    }
  }

  if (parseError) {
    result.warnings.push(parseError.message);
  }
  if (bufferBytes >= MAX_HEADER_READ_BYTES && bufferBytes < fileSizeBytes) {
    result.warnings.push(`GGUF metadata exceeded the ${Math.trunc(MAX_HEADER_READ_BYTES / 1024 / 1024)} MiB inspection cap.`);
  }
  result.metadataHash = hashJson({
    fileSizeBytes,
    quant: fallbackQuant,
    architecture: fallbackArchitecture,
    parserFallback: true,
  });
  return result;
}

export function deriveLoadingPolicy({
  gguf = null,
  hardware = null,
  estimate = null,
  requestedContextSize = 4096,
  requestedGpuLayers = "auto",
  mode = "generate",
}) {
  const fileSizeGb = gguf?.fileSizeGb ?? null;
  const ramBudgetGb = hardware?.memory?.plannerBudgetGB ?? hardware?.memory?.availableGB ?? null;
  const ramRequiredGb = estimate?.ramRequiredGB ?? estimate?.totalGB ?? null;
  const ramHeadroomGb = ramBudgetGb != null && ramRequiredGb != null ? ramBudgetGb - ramRequiredGb : null;
  const expectedLoadClass = fileSizeGb == null
    ? "medium"
    : fileSizeGb >= 6
      ? "slow"
      : fileSizeGb >= 2
        ? "medium"
        : "fast";
  const memoryRisk = ramRequiredGb == null || ramBudgetGb == null
    ? "medium"
    : ramRequiredGb <= ramBudgetGb * 0.8
      ? "low"
      : ramRequiredGb <= ramBudgetGb
        ? "medium"
        : "high";
  const metadataContextLimit = gguf?.contextLength ?? requestedContextSize;
  const recommendedContextSize = Math.max(512, Math.min(requestedContextSize, metadataContextLimit || requestedContextSize));

  return {
    mmap: true,
    mlock: Boolean(ramHeadroomGb != null && fileSizeGb != null && ramHeadroomGb >= Math.max(2, fileSizeGb * 0.75)),
    preload: false,
    warmupPrompt: ["benchmark", "pro", "server", "run"].includes(String(mode)),
    expectedLoadClass,
    memoryRisk,
    recommendedContextSize,
    recommendedGpuLayers: hardware?.primaryGpu?.vramGB ? requestedGpuLayers : 0,
    warnings: [
      ...(gguf?.status && gguf.status !== "ready" ? ["GGUF metadata is incomplete; using conservative loading defaults."] : []),
      ...(gguf?.warnings ?? []),
    ],
  };
}
