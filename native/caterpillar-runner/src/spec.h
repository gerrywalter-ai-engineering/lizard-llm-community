#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace cat {

struct NgramDraft {
    std::vector<std::uint32_t> tokens;
    std::size_t matched_tokens{};
};

[[nodiscard]] NgramDraft draft_from_ngram(
    std::span<const std::uint32_t> sequence,
    std::size_t max_draft_tokens,
    std::size_t min_match_tokens = 2,
    std::size_t max_match_tokens = 8
);

int run_spec_selftest();

}  // namespace cat
