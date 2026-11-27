// EnhancementRun_04 Phase 5 -- cross-session prefix cache.
//
// The runner already reuses a KV prefix, but only within one session's own
// history (`try_reuse_kv_prefix`). Eight chat sessions sharing a system prompt
// store it eight times, which is the single largest avoidable KV cost on an
// agent or support workload -- and on a 16 GB UMA box, KV footprint is what
// sets how large a batch can be, and batch size is the throughput multiplier.
//
// This maps content-addressed blocks to pool blocks so identical prefixes are
// stored once and shared by refcount.
//
// The hash is CHAINED: a block's hash folds in its predecessor's. A hit on
// block k therefore implies every block before it also matched, so a prefix can
// never be assembled out of blocks that happen to collide individually. That
// property is what makes "stop at the first miss" a correct rule rather than an
// approximation.
//
// Eviction is restricted to refcount 0. Evicting an in-use block is not a
// performance bug, it is a correctness bug that hands one session another's
// context and produces a fluent, wrong answer.

#ifndef LIZARD_NATIVE_PREFIX_CACHE_H
#define LIZARD_NATIVE_PREFIX_CACHE_H

#include <cstdint>
#include <cstddef>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#include "kv_blocks.h"

namespace lizard_native {

bool prefix_cache_enabled();

using PrefixHash = std::uint64_t;

// Folds one block's token ids into the running chain. `previous` is the hash of
// the preceding block, or 0 for the first.
PrefixHash hash_prefix_block(PrefixHash previous, const std::uint64_t *tokens, std::size_t count);

struct PrefixCacheStats {
  std::uint64_t hits = 0;
  std::uint64_t misses = 0;
  std::uint64_t insertions = 0;
  std::uint64_t evictions = 0;
  std::uint64_t evictions_refused = 0;  // wanted to evict, block still in use
  std::size_t entries = 0;
};

class PrefixCache {
 public:
  explicit PrefixCache(std::size_t capacity = 4096) : capacity_(capacity) {}

  void clear();
  PrefixCacheStats stats() const;

  // Number of whole blocks at the head of `tokens` that are already cached.
  // Stops at the first miss, which the chained hash makes sufficient.
  std::size_t match_length_blocks(
    const std::vector<std::uint64_t> &tokens,
    std::size_t block_tokens) const;

  // Looks up the block for a chain position, refreshing its LRU recency.
  // Returns k_invalid_kv_block on a miss.
  KvBlockId lookup(PrefixHash hash);

  // Records `block` under `hash`. The caller must already hold a reference;
  // the cache adds its own, so a cached block never drops to zero while listed.
  void insert(PrefixHash hash, KvBlockId block, KvBlockPool &pool);

  // Frees cache-held references until `needed` blocks are free, oldest first.
  // Blocks still referenced by a live sequence are skipped, not forced.
  // Returns how many were actually released.
  std::size_t evict_until_free(KvBlockPool &pool, std::size_t needed);

  bool validate(const KvBlockPool &pool, std::string *detail = nullptr) const;

 private:
  struct Entry {
    PrefixHash hash = 0;
    KvBlockId block = k_invalid_kv_block;
  };

  std::size_t capacity_;
  std::list<Entry> recency_;  // front = most recently used
  std::unordered_map<PrefixHash, std::list<Entry>::iterator> index_;
  mutable PrefixCacheStats stats_;
};

int prefix_cache_selftest();

}  // namespace lizard_native

#endif  // LIZARD_NATIVE_PREFIX_CACHE_H
