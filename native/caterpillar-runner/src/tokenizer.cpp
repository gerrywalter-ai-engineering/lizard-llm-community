#include "tokenizer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace cat {
namespace {

constexpr std::string_view k_space_marker = "\xE2\x96\x81";

bool is_byte_token(std::string_view token, unsigned char& value) {
    if (token.size() != 6 || !token.starts_with("<0x") || token.back() != '>') return false;
    unsigned int parsed{};
    std::istringstream input{std::string(token.substr(3, 2))};
    input >> std::hex >> parsed;
    if (!input || parsed > 255U) return false;
    value = static_cast<unsigned char>(parsed);
    return true;
}

std::string sentencepiece_input(std::string_view input, bool add_space_prefix) {
    std::string output;
    if (add_space_prefix) output.append(k_space_marker);
    output.reserve(input.size() + (add_space_prefix ? k_space_marker.size() : 0));
    for (const char value : input) {
        if (value == ' ') output.append(k_space_marker);
        else output.push_back(value);
    }
    return output;
}

std::string replace_space_markers(std::string_view input) {
    std::string output;
    for (std::size_t offset = 0; offset < input.size();) {
        if (input.substr(offset).starts_with(k_space_marker)) {
            output.push_back(' ');
            offset += k_space_marker.size();
        } else {
            output.push_back(input[offset++]);
        }
    }
    return output;
}

std::string utf8_from_codepoint(std::uint32_t codepoint) {
    std::string output;
    if (codepoint <= 0x7FU) output.push_back(static_cast<char>(codepoint));
    else if (codepoint <= 0x7FFU) {
        output.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint <= 0xFFFFU) {
        output.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
        output.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
    return output;
}

const std::array<std::string, 256>& gpt2_byte_symbols() {
    static const auto symbols = [] {
        std::array<std::string, 256> result;
        std::vector<int> bytes;
        std::vector<std::uint32_t> codepoints;
        for (int value = 33; value <= 126; ++value) { bytes.push_back(value); codepoints.push_back(value); }
        for (int value = 161; value <= 172; ++value) { bytes.push_back(value); codepoints.push_back(value); }
        for (int value = 174; value <= 255; ++value) { bytes.push_back(value); codepoints.push_back(value); }
        int extra = 0;
        for (int value = 0; value < 256; ++value) {
            if (std::find(bytes.begin(), bytes.end(), value) != bytes.end()) continue;
            bytes.push_back(value);
            codepoints.push_back(static_cast<std::uint32_t>(256 + extra++));
        }
        for (std::size_t index = 0; index < bytes.size(); ++index) result[bytes[index]] = utf8_from_codepoint(codepoints[index]);
        return result;
    }();
    return symbols;
}

const std::unordered_map<std::string, unsigned char>& gpt2_symbol_bytes() {
    static const auto inverse = [] {
        std::unordered_map<std::string, unsigned char> result;
        const auto& symbols = gpt2_byte_symbols();
        for (std::size_t byte = 0; byte < symbols.size(); ++byte) {
            result.emplace(symbols[byte], static_cast<unsigned char>(byte));
        }
        return result;
    }();
    return inverse;
}

void append_gpt2_decoded(std::string& output, std::string_view token) {
    const auto& inverse = gpt2_symbol_bytes();
    for (std::size_t offset = 0; offset < token.size();) {
        const auto lead = static_cast<unsigned char>(token[offset]);
        const std::size_t width = lead < 0x80U ? 1U
            : (lead & 0xE0U) == 0xC0U ? 2U
            : (lead & 0xF0U) == 0xE0U ? 3U
            : (lead & 0xF8U) == 0xF0U ? 4U
            : 1U;
        const auto available = std::min(width, token.size() - offset);
        const std::string symbol(token.substr(offset, available));
        if (const auto found = inverse.find(symbol); found != inverse.end()) {
            output.push_back(static_cast<char>(found->second));
        } else {
            output.append(symbol);
        }
        offset += available;
    }
}

}  // namespace

Tokenizer::Tokenizer(const GgufModelInfo& model) : model_(&model) {
    if (model.token_vocabulary.empty()) throw std::invalid_argument("GGUF has no tokenizer vocabulary");
    for (std::size_t index = 0; index < model.token_vocabulary.size(); ++index) {
        const auto& token = model.token_vocabulary[index];
        token_ids_.try_emplace(token, static_cast<std::uint32_t>(index));
        unsigned char byte{};
        if (is_byte_token(token, byte)) byte_tokens_.try_emplace(byte, static_cast<std::uint32_t>(index));
        if (!token.empty()) {
            const auto type = index < model.token_types.size() ? model.token_types[index] : 1;
            if (type == 1 || type == 4 || model.token_types.empty()) {
                prefix_tokens_[static_cast<unsigned char>(token.front())].push_back(static_cast<std::uint32_t>(index));
            }
        }
    }
    for (auto& [prefix, tokens] : prefix_tokens_) {
        static_cast<void>(prefix);
        std::stable_sort(tokens.begin(), tokens.end(), [&](std::uint32_t left, std::uint32_t right) {
            return model.token_vocabulary[left].size() > model.token_vocabulary[right].size();
        });
    }
    for (std::size_t index = 0; index < model.token_merges.size(); ++index) merge_ranks_.try_emplace(model.token_merges[index], index);
}

std::vector<std::uint32_t> Tokenizer::encode_sentencepiece(std::string_view text) const {
    const auto normalized = sentencepiece_input(text, model_->add_space_prefix);
    struct State { float score{-std::numeric_limits<float>::infinity()}; std::size_t previous{}; std::uint32_t token{}; };
    std::vector<State> states(normalized.size() + 1);
    states[0].score = 0.0F;
    for (std::size_t offset = 0; offset < normalized.size(); ++offset) {
        if (!std::isfinite(states[offset].score)) continue;
        const auto found = prefix_tokens_.find(static_cast<unsigned char>(normalized[offset]));
        if (found != prefix_tokens_.end()) {
            for (const auto id : found->second) {
                const auto& piece = model_->token_vocabulary[id];
                if (piece.empty() || offset + piece.size() > normalized.size()
                    || normalized.compare(offset, piece.size(), piece) != 0) continue;
                const float piece_score = id < model_->token_scores.size() ? model_->token_scores[id] : static_cast<float>(piece.size());
                auto& next = states[offset + piece.size()];
                const float candidate = states[offset].score + piece_score;
                if (candidate > next.score) next = {candidate, offset, id};
            }
        }
        const auto byte = static_cast<unsigned char>(normalized[offset]);
        if (const auto byte_id = byte_tokens_.find(byte); byte_id != byte_tokens_.end()) {
            auto& next = states[offset + 1];
            const float candidate = states[offset].score - 1000.0F;
            if (candidate > next.score) next = {candidate, offset, byte_id->second};
        }
    }
    if (!std::isfinite(states.back().score)) {
        throw std::runtime_error("SentencePiece vocabulary cannot represent the input bytes");
    }
    std::vector<std::uint32_t> output;
    for (std::size_t cursor = normalized.size(); cursor != 0; cursor = states[cursor].previous) output.push_back(states[cursor].token);
    std::reverse(output.begin(), output.end());
    return output;
}

std::vector<std::uint32_t> Tokenizer::encode_gpt2(std::string_view text) const {
    std::vector<std::string> symbols;
    symbols.reserve(text.size());
    for (const unsigned char byte : text) symbols.push_back(gpt2_byte_symbols()[byte]);
    while (symbols.size() > 1) {
        std::size_t best = symbols.size();
        std::size_t rank = std::numeric_limits<std::size_t>::max();
        for (std::size_t index = 0; index + 1 < symbols.size(); ++index) {
            const auto found = merge_ranks_.find(symbols[index] + " " + symbols[index + 1]);
            if (found != merge_ranks_.end() && found->second < rank) { rank = found->second; best = index; }
        }
        if (best == symbols.size()) break;
        symbols[best] += symbols[best + 1];
        symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best + 1));
    }
    std::vector<std::uint32_t> output;
    for (const auto& symbol : symbols) {
        const auto found = token_ids_.find(symbol);
        if (found == token_ids_.end()) throw std::runtime_error("GPT-2 BPE vocabulary cannot represent the input");
        output.push_back(found->second);
    }
    return output;
}

std::vector<std::uint32_t> Tokenizer::encode(std::string_view text, bool add_special) const {
    std::vector<std::uint32_t> output;
    std::string model = model_->tokenizer_model;
    std::transform(model.begin(), model.end(), model.begin(), [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    if (model == "gpt2") output = encode_gpt2(text);
    else output = encode_sentencepiece(text);
    if (add_special && model_->add_bos_token && (output.empty() || output.front() != model_->bos_token_id)) {
        output.insert(output.begin(), static_cast<std::uint32_t>(model_->bos_token_id));
    }
    if (add_special && model_->add_eos_token && (output.empty() || output.back() != model_->eos_token_id)) {
        output.push_back(static_cast<std::uint32_t>(model_->eos_token_id));
    }
    return output;
}

std::string Tokenizer::token_text(std::uint32_t token) const {
    if (token >= model_->token_vocabulary.size()) throw std::out_of_range("token id is outside the vocabulary");
    return model_->token_vocabulary[token];
}

std::string Tokenizer::decode(const std::vector<std::uint32_t>& tokens, bool skip_special) const {
    std::string output;
    std::string tokenizer_model = model_->tokenizer_model;
    std::transform(tokenizer_model.begin(), tokenizer_model.end(), tokenizer_model.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    const bool gpt2 = tokenizer_model == "gpt2";
    for (const auto id : tokens) {
        if (id >= model_->token_vocabulary.size()) throw std::out_of_range("token id is outside the vocabulary");
        const auto type = id < model_->token_types.size() ? model_->token_types[id] : 1;
        if (skip_special && (type == 3 || id == model_->bos_token_id || id == model_->eos_token_id || id == model_->padding_token_id)) continue;
        const auto& token = model_->token_vocabulary[id];
        unsigned char byte{};
        if (is_byte_token(token, byte)) output.push_back(static_cast<char>(byte));
        else if (gpt2) append_gpt2_decoded(output, token);
        else output += replace_space_markers(token);
    }
    if (!gpt2 && !output.empty() && output.front() == ' ') output.erase(output.begin());
    return output;
}

std::uint32_t greedy_argmax(const std::vector<float>& logits) {
    if (logits.empty()) throw std::invalid_argument("cannot sample empty logits");
    std::uint32_t best = 0;
    for (std::size_t index = 1; index < logits.size(); ++index) {
        if (std::isnan(logits[index])) continue;
        if (std::isnan(logits[best]) || logits[index] > logits[best]) best = static_cast<std::uint32_t>(index);
    }
    return best;
}

int run_tokenizer_selftest() {
    try {
        GgufModelInfo model;
        model.tokenizer_model = "llama";
        model.bos_token_id = 0;
        model.eos_token_id = 1;
        model.add_bos_token = true;
        model.add_space_prefix = true;
        model.token_vocabulary = {"<s>", "</s>", "<unk>", "\xE2\x96\x81", "H", "e", "l", "o", "\xE2\x96\x81Hello", "!", "<0x3F>"};
        model.token_scores = {0, 0, -100, -10, -5, -5, -5, -5, -1, -2, -20};
        model.token_types = {3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 6};
        Tokenizer tokenizer(model);
        const auto encoded = tokenizer.encode("Hello!", true);
        if (encoded != std::vector<std::uint32_t>({0, 8, 9})) throw std::runtime_error("SentencePiece encode mismatch");
        if (tokenizer.decode(encoded) != "Hello!") throw std::runtime_error("SentencePiece decode mismatch");
        GgufModelInfo gpt2_model;
        gpt2_model.tokenizer_model = "gpt2";
        gpt2_model.add_bos_token = false;
        gpt2_model.bos_token_id = 99;
        gpt2_model.eos_token_id = 98;
        gpt2_model.padding_token_id = 97;
        gpt2_model.token_vocabulary = {"!", "\xC4\xA0" "anytime"};
        Tokenizer gpt2_tokenizer(gpt2_model);
        if (gpt2_tokenizer.decode({1}) != " anytime") throw std::runtime_error("GPT-2 byte decode mismatch");
        if (greedy_argmax({1.0F, 3.0F, 3.0F, 2.0F}) != 1) throw std::runtime_error("greedy tie-breaking mismatch");
        std::cout << "caterpillar tokenizer/sampler selftest: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "caterpillar tokenizer/sampler selftest: failed: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace cat
