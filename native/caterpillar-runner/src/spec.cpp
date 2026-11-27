#include "spec.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace cat {

NgramDraft draft_from_ngram(
    std::span<const std::uint32_t> sequence,
    std::size_t max_draft_tokens,
    std::size_t min_match_tokens,
    std::size_t max_match_tokens
) {
    NgramDraft result;
    if (max_draft_tokens == 0 || min_match_tokens == 0
        || sequence.size() <= min_match_tokens) {
        return result;
    }
    const auto longest = std::min(max_match_tokens, sequence.size() - 1);
    for (std::size_t match = longest; match >= min_match_tokens; --match) {
        const auto suffix = sequence.last(match);
        // Prefer the most recent occurrence. It tends to track local loops and
        // makes the draft deterministic when the same suffix appeared often.
        for (std::size_t position = sequence.size() - match; position-- > 0;) {
            if (position + match >= sequence.size()) continue;
            if (!std::equal(suffix.begin(), suffix.end(), sequence.begin() + static_cast<std::ptrdiff_t>(position))) {
                continue;
            }
            const auto continuation = position + match;
            const auto available = sequence.size() - continuation;
            const auto count = std::min(max_draft_tokens, available);
            if (count == 0) continue;
            result.matched_tokens = match;
            result.tokens.assign(
                sequence.begin() + static_cast<std::ptrdiff_t>(continuation),
                sequence.begin() + static_cast<std::ptrdiff_t>(continuation + count));
            return result;
        }
        if (match == min_match_tokens) break;
    }
    return result;
}

int run_spec_selftest() {
    try {
        const std::vector<std::uint32_t> repeating{1, 2, 3, 1, 2, 3, 1, 2, 3};
        const auto draft = draft_from_ngram(repeating, 4);
        if (draft.matched_tokens != 6 || draft.tokens != std::vector<std::uint32_t>({1, 2, 3})) {
            throw std::runtime_error("longest-suffix n-gram draft mismatch");
        }
        const std::vector<std::uint32_t> ambiguous{4, 5, 9, 4, 5, 8, 4, 5};
        const auto recent = draft_from_ngram(ambiguous, 1, 2, 2);
        if (recent.tokens != std::vector<std::uint32_t>({8})) {
            throw std::runtime_error("n-gram draft did not prefer the most recent continuation");
        }
        if (!draft_from_ngram(ambiguous, 1, 4, 8).tokens.empty()) {
            throw std::runtime_error("high-confidence n-gram gate accepted a short suffix");
        }
        const std::vector<std::uint32_t> unique{1, 2, 3, 4};
        if (!draft_from_ngram(unique, 4).tokens.empty()) {
            throw std::runtime_error("unique context unexpectedly produced a draft");
        }
        std::cout << "caterpillar speculative draft selftest: ok (longest suffix + recent continuation + no-match)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "caterpillar speculative draft selftest: failed: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace cat
