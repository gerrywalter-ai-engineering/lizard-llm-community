// EnhancementRun_04 Phase 3 -- paged KV.
//
// Today `SessionState::layered_kv_cache` is a contiguous append-only lane per
// session per layer. Two consequences decide the whole campaign:
//
//   1. Every live session reserves for its MAXIMUM context, not its current
//      length, so B sequences cost B full reservations.
//   2. Nothing can be shared, so an identical system prompt is stored B times.
//
// On a 16 GB UMA box that is what caps B at 1-2 before any batching code
// exists. This file replaces the reservation model with fixed-size blocks and
// a per-sequence block table -- the allocation shape PagedAttention introduced
// and the reason vLLM can hold many sequences in the memory one would need.
//
// DELIBERATELY BYTE-ORIENTED. The pool never interprets what a row contains,
// so it works unchanged across f32/f16/q8_0/q4_0/q3_wht: only `row_bytes`
// changes. Putting float semantics in here would have forced a second pool per
// KV mode and coupled paging to quantization, which are orthogonal.
//
// The failure mode this file is written against is not a crash. A wrong block
// table produces fluent, confident, wrong text and never faults, so every
// mutation validates before it writes and the accounting is fuzz-tested.

#ifndef LIZARD_NATIVE_KV_BLOCKS_H
#define LIZARD_NATIVE_KV_BLOCKS_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lizard_native {

using KvBlockId = std::uint32_t;

inline constexpr KvBlockId k_invalid_kv_block = static_cast<KvBlockId>(-1);

bool kv_paged_enabled();
std::size_t kv_block_tokens();

struct KvBlockPoolConfig {
  std::size_t block_tokens = 16;   // token positions per block
  std::size_t row_bytes = 0;       // bytes for one position's payload
  std::size_t block_count = 0;     // blocks in the arena
  std::size_t reserve_blocks = 0;  // floor kept free so running sequences grow
};

// Per-sequence view: logical position i lives in block_ids[i / block_tokens]
// at slot i % block_tokens.
class KvSequence {
 public:
  std::size_t length() const { return length_; }
  std::size_t shared_prefix() const { return shared_prefix_; }
  const std::vector<KvBlockId> &block_ids() const { return block_ids_; }

 private:
  friend class KvBlockPool;
  std::vector<KvBlockId> block_ids_;
  std::size_t length_ = 0;
  std::size_t shared_prefix_ = 0;
};

enum class KvAdmission {
  Admitted,
  Wait,        // pool pressure; caller should retry after a retirement
  TooLarge,    // larger than the whole pool; can never succeed
};

struct KvPoolStats {
  std::size_t total_blocks = 0;
  std::size_t used_blocks = 0;
  std::size_t free_blocks = 0;
  std::size_t shared_blocks = 0;      // refcount > 1
  std::uint64_t copy_on_writes = 0;
  std::uint64_t allocations = 0;
  std::uint64_t releases = 0;
};

class KvBlockPool {
 public:
  KvBlockPool() = default;
  explicit KvBlockPool(const KvBlockPoolConfig &config) { reset(config); }

  void reset(const KvBlockPoolConfig &config);

  const KvBlockPoolConfig &config() const { return config_; }
  KvPoolStats stats() const;

  // How many blocks a sequence of `tokens` positions needs.
  std::size_t blocks_for_tokens(std::size_t tokens) const;

  // Admission is decided ONCE, before a sequence starts, against its whole
  // projected length. There is deliberately no mid-decode allocation failure
  // path: a sequence that was admitted must be able to finish, because the
  // alternative is discovering the pool is empty halfway through a token.
  KvAdmission can_admit(std::size_t projected_tokens) const;

  // Appends one position's payload. Returns false when the pool is exhausted,
  // which admission is supposed to have made impossible -- callers treat it as
  // a bug, not a condition.
  bool append_row(KvSequence &sequence, const std::uint8_t *row, std::size_t row_bytes);

  // Reads one position. Returns nullptr for an out-of-range position rather
  // than reading adjacent memory, because a silently wrong row is the exact
  // failure this design is guarding against.
  const std::uint8_t *row(const KvSequence &sequence, std::size_t position) const;

  // Drops everything past `keep_positions`, releasing whole blocks that fall
  // entirely outside it. Mirrors the existing prefix-reuse rewind.
  void truncate(KvSequence &sequence, std::size_t keep_positions);

  void release(KvSequence &sequence);

  // Shares the first `positions` of `source` into `target`, bumping refcounts.
  // Only whole blocks can be shared: a partially-filled block would let the new
  // sequence's writes land in the donor's tail. Returns the number of positions
  // actually shared, which is `positions` rounded DOWN to a block boundary.
  std::size_t share_prefix(const KvSequence &source, KvSequence &target, std::size_t positions);

  // Copy-on-write: gives `sequence` a private copy of the block holding
  // `position` when that block is shared. Called before any write to a block
  // whose refcount exceeds one.
  bool make_writable(KvSequence &sequence, std::size_t position);

  // Invariant check for tests and for the fuzz harness: used + free == total,
  // and every block a sequence references has a non-zero refcount.
  bool validate(std::string *detail = nullptr) const;

  // --- prefix-cache support ------------------------------------------------
  // The cache holds its own reference to every block it lists, so a block
  // cannot be freed by the sequence that created it while still cached. These
  // are the only ways to take and drop that reference; nothing else in the
  // runner manipulates refcounts directly.
  std::uint16_t block_refcount(KvBlockId block) const;
  void retain_block(KvBlockId block);
  void release_cached_block(KvBlockId block);

  // Allocates a bare block. Exposed for the selftests, which need to populate
  // a cache without building a whole sequence around every entry.
  KvBlockId allocate_for_test() { return allocate_block(); }

 private:
  KvBlockId allocate_block();
  void release_block(KvBlockId block);
  std::uint8_t *block_data(KvBlockId block);
  const std::uint8_t *block_data(KvBlockId block) const;

  KvBlockPoolConfig config_;
  std::vector<std::uint8_t> arena_;
  std::vector<KvBlockId> free_list_;
  std::vector<std::uint16_t> refcount_;
  std::uint64_t copy_on_writes_ = 0;
  std::uint64_t allocations_ = 0;
  std::uint64_t releases_ = 0;
};

// Runs the paged-KV correctness suite. Returns 0 on success and prints one
// line per case. Exposed as `lizard-native-runner --kv-blocks-selftest` so the
// JS test suite can gate on it without linking C++.
//
// This is the phase's gate, and it is deliberately not "the generated text
// looks right": a wrong block table produces fluent, confident, wrong output
// and never faults, so the checks are bit-identity against a contiguous
// reference plus exhaustive accounting.
int kv_blocks_selftest();

}  // namespace lizard_native

#endif  // LIZARD_NATIVE_KV_BLOCKS_H
