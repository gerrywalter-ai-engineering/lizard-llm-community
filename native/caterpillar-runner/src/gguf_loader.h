#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace cat {

struct GgufTensorInfo {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    std::uint32_t type{};
    std::uint64_t offset{};
};

struct GgufModelInfo {
    std::uint32_t version{};
    std::uint64_t metadata_count{};
    std::string name;
    std::string architecture;
    std::string family;
    std::string quantization{"UNKNOWN"};
    std::uint64_t block_count{};
    std::uint64_t embedding_length{};
    std::uint64_t feed_forward_length{};
    std::uint64_t head_count{};
    std::uint64_t head_count_kv{};
    std::uint64_t rope_dimension_count{};
    // Gemma 3 sets attention.key_length to 256 while hidden/heads is 320, so
    // the head dimension cannot be derived from the hidden size for every
    // architecture. Zero when the file does not state one.
    std::uint64_t key_length{};
    std::uint64_t value_length{};
    std::uint64_t context_length{};
    std::uint64_t attention_key_length{};
    std::uint64_t attention_value_length{};
    std::uint64_t attention_key_length_swa{};
    std::uint64_t attention_value_length_swa{};
    std::uint64_t attention_sliding_window{};
    std::uint64_t attention_shared_kv_layers{};
    std::uint64_t embedding_length_per_layer_input{};
    std::uint64_t vocabulary_size{};
    std::string tokenizer_model;
    std::vector<std::string> token_vocabulary;
    std::vector<float> token_scores;
    std::vector<std::int32_t> token_types;
    std::vector<std::string> token_merges;
    std::uint64_t bos_token_id{};
    std::uint64_t eos_token_id{};
    std::uint64_t unknown_token_id{};
    std::uint64_t padding_token_id{};
    bool add_bos_token{true};
    bool add_eos_token{false};
    bool add_space_prefix{true};
    float rope_frequency_base{10000.0F};
    float rope_frequency_base_swa{10000.0F};
    float rope_frequency_scale{1.0F};
    float rms_epsilon{1.0e-5F};
    float final_logit_softcap{};
    std::uint32_t alignment{32};
    std::uint64_t tensor_data_offset{};
    std::vector<GgufTensorInfo> tensors;
    std::vector<std::uint64_t> missing_attention_value_layers;

    [[nodiscard]] bool tensor_directory_complete() const noexcept {
        return missing_attention_value_layers.empty();
    }
};

[[nodiscard]] GgufModelInfo inspect_gguf(const std::filesystem::path& path);
[[nodiscard]] std::string gguf_info_json(const GgufModelInfo& info);
int run_gguf_selftest();

}  // namespace cat
