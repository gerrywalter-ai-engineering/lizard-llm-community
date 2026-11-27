import fs from "node:fs";
import path from "node:path";

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

export function writeOneLayerGguf(filePath) {
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

function kvFloat32(key, value) {
  return Buffer.concat([encodeString(key), encodeUint32(6), encodeFloat32(value)]);
}

// Deterministic pseudo-random weights so every family fixture is stable
// across runs without hand-tuning per-tensor values.
function seededWeights(count, seed) {
  const values = [];
  let state = seed >>> 0;
  for (let index = 0; index < count; index += 1) {
    state = (state * 1664525 + 1013904223) >>> 0;
    values.push(((state / 0xffffffff) - 0.5) * 0.8);
  }
  return values;
}

// One-layer (two for deepseek) synthetic GGUF for each native-worker family
// group. Dimensions: hidden=2, heads=1, kv-heads=1, ff=2, vocab=5, f32-only.
export function writeFamilyGguf(filePath, family) {
  const layouts = {
    qwen2: { arch: "qwen2", qkvBias: true },
    qwen3: { arch: "qwen3", qkNorm: true, keyLength: 2 },
    mistral: { arch: "mistral" },
    mixtral: { arch: "llama", moe: true },
    phi3: { arch: "phi3", fusedQkv: true, fusedFfn: true, ropeDim: 2 },
    phi: { arch: "phi2", layerNorm: true, geluFfn: true, parallel: true, denseBias: true, outputBias: true, ropeDim: 2 },
    glm: { arch: "glm4", fusedFfn: true, postNorms: true, ropeDim: 2 },
    starcoder2: { arch: "starcoder2", layerNorm: true, geluFfn: true, denseBias: true },
    qwen2moe: { arch: "qwen2moe", moe: true, sharedExpert: true, sharedExpertGated: true },
    qwen3moe: { arch: "qwen3moe", qkNorm: true, keyLength: 2, moe: true },
    deepseek: { arch: "deepseek", moe: true, sharedExpert: true, denseLead: 1, blocks: 2 },
    // Blocked-family probe: same tensors as deepseek but the deepseek2 arch,
    // which requires MLA attention the worker does not implement yet.
    deepseek2: { arch: "deepseek2", moe: true, sharedExpert: true, denseLead: 1, blocks: 2 },
  };
  const layout = layouts[family];
  if (!layout) {
    throw new Error(`Unknown family fixture '${family}'.`);
  }
  const arch = layout.arch;
  const blocks = layout.blocks ?? 1;

  const metadata = [
    kvString("general.architecture", arch),
    kvUint32("general.file_type", 0),
    kvString("tokenizer.ggml.model", "synthetic-greedy"),
    kvUint32(`${arch}.context_length`, 32),
    kvUint32(`${arch}.embedding_length`, 2),
    kvUint32(`${arch}.block_count`, blocks),
    kvUint32(`${arch}.attention.head_count`, 1),
    kvUint32(`${arch}.attention.head_count_kv`, 1),
    kvFloat32(`${arch}.rope.freq_base`, 10000),
    kvString("tokenizer.chat_template", "{{ prompt }}"),
    kvUint32("tokenizer.ggml.bos_token_id", 0),
    kvUint32("tokenizer.ggml.eos_token_id", 4),
    kvStringArray("tokenizer.ggml.tokens", ["<s>", "hello", " ", "world", "</s>"]),
  ];
  if (layout.keyLength) {
    metadata.push(kvUint32(`${arch}.attention.key_length`, layout.keyLength));
  }
  if (layout.ropeDim) {
    metadata.push(kvUint32(`${arch}.rope.dimension_count`, layout.ropeDim));
  }
  if (layout.layerNorm) {
    metadata.push(kvFloat32(`${arch}.attention.layer_norm_epsilon`, 1e-5));
  } else {
    metadata.push(kvFloat32(`${arch}.attention.layer_norm_rms_epsilon`, 1e-5));
  }
  if (layout.moe) {
    metadata.push(kvUint32(`${arch}.expert_count`, 2));
    metadata.push(kvUint32(`${arch}.expert_used_count`, 2));
    metadata.push(kvUint32(`${arch}.expert_feed_forward_length`, 2));
  }
  if (layout.denseLead) {
    metadata.push(kvUint32(`${arch}.leading_dense_block_count`, layout.denseLead));
  }

  const tensors = [];
  const tensorData = [];
  let offsetBytes = 0;
  let seed = 7;
  const pushTensor = (name, dimensions, values = null) => {
    const elementCount = dimensions.reduce((product, dimension) => product * dimension, 1);
    const payload = encodeFloatTensor(values ?? seededWeights(elementCount, (seed += 13)));
    tensors.push(tensorEntry(name, dimensions, 0, offsetBytes));
    tensorData.push(payload);
    offsetBytes += payload.length;
  };
  const ones = (count) => Array.from({ length: count }, () => 1);
  const smallBias = (count) => Array.from({ length: count }, () => 0.05);

  pushTensor("token_embd.weight", [2, 5], [
    0.2, -0.1,
    1, 0,
    0, 1,
    1, 1,
    -1, 0.3,
  ]);
  for (let block = 0; block < blocks; block += 1) {
    const prefix = `blk.${block}.`;
    pushTensor(`${prefix}attn_norm.weight`, [2], ones(2));
    if (layout.layerNorm) {
      pushTensor(`${prefix}attn_norm.bias`, [2], smallBias(2));
    }
    if (layout.fusedQkv) {
      pushTensor(`${prefix}attn_qkv.weight`, [2, 6]);
    } else {
      pushTensor(`${prefix}attn_q.weight`, [2, 2]);
      pushTensor(`${prefix}attn_k.weight`, [2, 2]);
      pushTensor(`${prefix}attn_v.weight`, [2, 2]);
      if (layout.qkvBias || layout.denseBias) {
        pushTensor(`${prefix}attn_q.bias`, [2], smallBias(2));
        pushTensor(`${prefix}attn_k.bias`, [2], smallBias(2));
        pushTensor(`${prefix}attn_v.bias`, [2], smallBias(2));
      }
    }
    pushTensor(`${prefix}attn_output.weight`, [2, 2]);
    if (layout.denseBias) {
      pushTensor(`${prefix}attn_output.bias`, [2], smallBias(2));
    }
    if (layout.qkNorm) {
      pushTensor(`${prefix}attn_q_norm.weight`, [2], ones(2));
      pushTensor(`${prefix}attn_k_norm.weight`, [2], ones(2));
    }
    if (layout.postNorms) {
      pushTensor(`${prefix}post_attention_norm.weight`, [2], ones(2));
      pushTensor(`${prefix}post_ffw_norm.weight`, [2], ones(2));
    }
    if (!layout.parallel) {
      pushTensor(`${prefix}ffn_norm.weight`, [2], ones(2));
      if (layout.layerNorm) {
        pushTensor(`${prefix}ffn_norm.bias`, [2], smallBias(2));
      }
    }
    const moeBlock = layout.moe && block >= (layout.denseLead ?? 0);
    if (moeBlock) {
      pushTensor(`${prefix}ffn_gate_inp.weight`, [2, 2]);
      pushTensor(`${prefix}ffn_gate_exps.weight`, [2, 2, 2]);
      pushTensor(`${prefix}ffn_up_exps.weight`, [2, 2, 2]);
      pushTensor(`${prefix}ffn_down_exps.weight`, [2, 2, 2]);
      if (layout.sharedExpert) {
        if (layout.sharedExpertGated) {
          pushTensor(`${prefix}ffn_gate_inp_shexp.weight`, [2]);
        }
        pushTensor(`${prefix}ffn_gate_shexp.weight`, [2, 2]);
        pushTensor(`${prefix}ffn_up_shexp.weight`, [2, 2]);
        pushTensor(`${prefix}ffn_down_shexp.weight`, [2, 2]);
      }
    } else {
      if (layout.fusedFfn) {
        pushTensor(`${prefix}ffn_up.weight`, [2, 4]);
      } else {
        if (!layout.geluFfn) {
          pushTensor(`${prefix}ffn_gate.weight`, [2, 2]);
        }
        pushTensor(`${prefix}ffn_up.weight`, [2, 2]);
        if (layout.denseBias) {
          pushTensor(`${prefix}ffn_up.bias`, [2], smallBias(2));
        }
      }
      pushTensor(`${prefix}ffn_down.weight`, [2, 2]);
      if (layout.denseBias) {
        pushTensor(`${prefix}ffn_down.bias`, [2], smallBias(2));
      }
    }
  }
  pushTensor("output_norm.weight", [2], ones(2));
  if (layout.layerNorm) {
    pushTensor("output_norm.bias", [2], smallBias(2));
  }
  pushTensor("output.weight", [2, 5], [
    0, 0.1,
    0.2, 0,
    0, 0.3,
    1.4, 0.2,
    0.9, 1,
  ]);
  if (layout.outputBias) {
    pushTensor("output.bias", [5], smallBias(5));
  }

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

export const NATIVE_FAMILY_FIXTURES = [
  "qwen2",
  "qwen3",
  "mistral",
  "mixtral",
  "phi",
  "phi3",
  "glm",
  "starcoder2",
  "qwen2moe",
  "qwen3moe",
  "deepseek",
];

export function compiledWorkerPath() {
  const candidates = [
    path.resolve(process.cwd(), "native", "lizard-native-runner", "build", "lizard-native-runner.exe"),
    path.resolve(process.cwd(), "native", "lizard-native-runner", "build-cmake", "lizard-native-runner.exe"),
  ];
  return candidates.find((candidate) => fs.existsSync(candidate)) ?? null;
}
