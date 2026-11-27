#pragma once

#include "family_spec.h"
#include "mapped_model.h"
#include "tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cat {

struct CpuGenerationResult {
    std::vector<std::uint32_t> prompt_tokens;
    std::vector<std::uint32_t> generated_tokens;
    std::string text;
};

struct CpuKvLayer {
    std::size_t layer{};
    std::size_t width{};
    std::vector<float> keys;
    std::vector<float> values;
};

struct CpuSessionState {
    std::size_t cached_tokens{};
    std::vector<CpuKvLayer> layers;
    std::uint32_t next_token{};
    bool next_token_valid{false};

    [[nodiscard]] std::size_t bytes() const noexcept;
};

struct CpuDraftVerification {
    std::vector<std::uint32_t> next_tokens;
    std::vector<CpuKvLayer> layers;
};

class CpuModel {
public:
    explicit CpuModel(const std::filesystem::path& path);

    [[nodiscard]] const GgufModelInfo& info() const noexcept { return model_.info(); }
    [[nodiscard]] const MappedGgufModel& mapped() const noexcept { return model_; }
    [[nodiscard]] const Tokenizer& tokenizer() const noexcept { return tokenizer_; }
    [[nodiscard]] std::vector<float> hidden_states(
        const std::vector<std::uint32_t>& tokens,
        std::size_t layer_limit = 0
    ) const;
    [[nodiscard]] std::uint32_t greedy_next(
        const std::vector<std::uint32_t>& tokens,
        std::size_t layer_limit = 0
    ) const;
    [[nodiscard]] CpuSessionState prefill(const std::vector<std::uint32_t>& tokens) const;
    [[nodiscard]] std::uint32_t advance(CpuSessionState& session, std::uint32_t token) const;
    [[nodiscard]] std::vector<std::uint32_t> advance_batch(
        std::span<CpuSessionState*> sessions,
        std::span<const std::uint32_t> tokens
    ) const;
    [[nodiscard]] CpuDraftVerification verify_draft(
        const CpuSessionState& session,
        const std::vector<std::uint32_t>& input_tokens
    ) const;
    // Phase 6 (2026-07-21 campaign): commit the accepted prefix of a draft
    // verification to the session - append the first committed_inputs
    // positions' K/V rows per layer, advance cached_tokens, and hand the
    // model's continuation token to the loop. The CPU mirror of the GPU
    // commit_draft contract: only verified-exact tokens are ever committed.
    void commit_draft(
        CpuSessionState& session,
        const CpuDraftVerification& verification,
        std::size_t committed_inputs
    ) const;
    [[nodiscard]] CpuGenerationResult generate(std::string_view prompt, std::size_t max_tokens) const;
    [[nodiscard]] std::vector<std::vector<float>> build_ple_inputs(
        const std::vector<std::uint32_t>& tokens,
        const std::vector<float>& scaled_embeddings
    ) const;
    [[nodiscard]] std::uint32_t greedy_from_hidden(std::span<const float> hidden) const;

private:
    struct InferenceResult {
        std::vector<float> hidden;
        std::vector<CpuKvLayer> layers;
    };

    // Phase 1 of .agent/vision/01_native_vision_plan.md, caterpillar side.
    //
    // Positions whose hidden state the caller supplies instead of looking up
    // from token_embd.weight. An image becomes a run of rows out of a vision
    // tower, and those rows have no token ids, so the prompt has to be able to
    // carry hidden states as well as ids.
    //
    // A supplied row is the FINAL hidden state and is used as given - notably
    // WITHOUT the gemma4 sqrt(hidden_size) scale, which belongs to looked-up
    // token embeddings only. A projector output is already in the model's
    // space; scaling it again would quietly corrupt it.
    struct EmbeddingOverrides {
        std::unordered_map<std::size_t, std::vector<float>> rows;

        [[nodiscard]] const std::vector<float>* row_for(std::size_t position) const {
            const auto found = rows.find(position);
            return found == rows.end() ? nullptr : &found->second;
        }
    };

    [[nodiscard]] InferenceResult infer(
        const std::vector<std::uint32_t>& tokens,
        const CpuSessionState* past,
        std::size_t layer_limit = 0,
        const EmbeddingOverrides* embedding_overrides = nullptr,
        std::span<CpuSessionState* const> independent_past = {}
    ) const;

    MappedGgufModel model_;
    Tokenizer tokenizer_;
    // Resolved once at construction: a pure function of the model metadata
    // that infer() used to recompute on every token. Copied, not referenced,
    // by infer() because a layer_limit request mutates its copy.
    FamilySpec base_family_;
};

}  // namespace cat
