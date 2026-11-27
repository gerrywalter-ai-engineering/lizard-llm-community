# Caterpillar runner

This is the standalone Caterpillar native module. Phase 0 established the
provider and stdio lifecycle. Phase 1 replaces the GGUF reference delegation
with Caterpillar's self-contained tokenizer, mapped-weight CPU graph, greedy
sampler, and worker RPC lifecycle. Prototype JSON models still use the Phase 0
reference path so existing non-GGUF tests remain compatible.

Phase 1 now contains the first self-contained CPU execution core:
`graph.*` validates a typed DAG, `alloc.*` plans one lifetime-reused activation
arena with safe in-place operations, and `plan.*` compiles/replays coalesced CPU
spans. `--graph-selftest` compares compiled replay against the imperative
reference bit-for-bit for both decode and prefill shapes.

`gguf_loader.*` is a clean-room GGUF v2/v3 metadata and tensor-directory
reader. It resolves the initial Llama/Gemma/Qwen family identity, quantization,
shape metadata, and the Gemma cross-layer attention-value preflight without
loading tokenizer arrays or model weights. Use `--inspect-gguf <path>` for its
machine-readable summary.

`family_spec.*` resolves model dimensions and emits a complete decoder-block
DAG for the initial Llama, Gemma, and Qwen2 families. The CPU plan includes
RMSNorm, Q/K/V projections, RoPE, grouped-query causal attention, attention
output/residual, SwiGLU FFN, and final normalization. The FamilySpec self-test
runs each family in decode and prefill shapes and requires bit-identical output
from imperative execution and compiled-plan replay.

No source, header, shader, or translation unit from llama.cpp, ggml,
ik_llama.cpp, or `native/lizard-native-runner` is included in this target.
