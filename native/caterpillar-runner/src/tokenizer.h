#pragma once

#include "gguf_loader.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cat {

class Tokenizer {
public:
    explicit Tokenizer(const GgufModelInfo& model);

    [[nodiscard]] std::vector<std::uint32_t> encode(std::string_view text, bool add_special = true) const;
    [[nodiscard]] std::string decode(const std::vector<std::uint32_t>& tokens, bool skip_special = true) const;
    [[nodiscard]] std::string token_text(std::uint32_t token) const;

private:
    [[nodiscard]] std::vector<std::uint32_t> encode_sentencepiece(std::string_view text) const;
    [[nodiscard]] std::vector<std::uint32_t> encode_gpt2(std::string_view text) const;

    const GgufModelInfo* model_{};
    std::unordered_map<std::string, std::uint32_t> token_ids_;
    std::unordered_map<std::string, std::size_t> merge_ranks_;
    std::unordered_map<unsigned char, std::vector<std::uint32_t>> prefix_tokens_;
    std::unordered_map<unsigned char, std::uint32_t> byte_tokens_;
};

[[nodiscard]] std::uint32_t greedy_argmax(const std::vector<float>& logits);
int run_tokenizer_selftest();

}  // namespace cat
