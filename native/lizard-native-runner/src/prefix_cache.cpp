#include "prefix_cache.h"

#include <cstdio>
#include <cstdlib>

namespace lizard_native {

bool prefix_cache_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("LIZARD_NATIVE_PREFIX_CACHE");
    return value != nullptr && std::string(value) == "1";
  }();
  return enabled;
}

PrefixHash hash_prefix_block(PrefixHash previous, const std::uint64_t *tokens, std::size_t count) {
  // FNV-1a over the token ids, seeded with the predecessor's hash so the value
  // identifies the whole prefix rather than this block alone. Two prefixes that
  // differ anywhere earlier cannot land on the same chain value except by a
  // full 64-bit collision.
  std::uint64_t hash = previous ^ 0xcbf29ce484222325ULL;
  for (std::size_t index = 0; index < count; index += 1) {
    const std::uint64_t token = tokens[index];
    for (int byte = 0; byte < 8; byte += 1) {
      hash ^= (token >> (byte * 8)) & 0xFF;
      hash *= 0x100000001b3ULL;
    }
  }
  // A zero hash would be indistinguishable from "no predecessor" at the head of
  // the chain, so it is nudged rather than allowed to alias.
  return hash == 0 ? 0x9e3779b97f4a7c15ULL : hash;
}

void PrefixCache::clear() {
  recency_.clear();
  index_.clear();
  stats_.entries = 0;
}

PrefixCacheStats PrefixCache::stats() const {
  stats_.entries = index_.size();
  return stats_;
}

std::size_t PrefixCache::match_length_blocks(
  const std::vector<std::uint64_t> &tokens,
  std::size_t block_tokens) const {
  if (block_tokens == 0) {
    return 0;
  }
  const std::size_t whole_blocks = tokens.size() / block_tokens;
  PrefixHash chain = 0;
  std::size_t matched = 0;
  for (std::size_t block = 0; block < whole_blocks; block += 1) {
    chain = hash_prefix_block(chain, tokens.data() + (block * block_tokens), block_tokens);
    if (index_.find(chain) == index_.end()) {
      break;  // chained hash: a miss here means nothing later can match either
    }
    matched += 1;
  }
  return matched;
}

KvBlockId PrefixCache::lookup(PrefixHash hash) {
  const auto found = index_.find(hash);
  if (found == index_.end()) {
    stats_.misses += 1;
    return k_invalid_kv_block;
  }
  stats_.hits += 1;
  recency_.splice(recency_.begin(), recency_, found->second);
  return found->second->block;
}

void PrefixCache::insert(PrefixHash hash, KvBlockId block, KvBlockPool &pool) {
  if (block == k_invalid_kv_block) {
    return;
  }
  const auto existing = index_.find(hash);
  if (existing != index_.end()) {
    recency_.splice(recency_.begin(), recency_, existing->second);
    return;  // already cached; do not take a second reference for one entry
  }
  // The cache holds its own reference so a listed block cannot be freed by the
  // sequence that created it.
  pool.retain_block(block);
  recency_.push_front(Entry {hash, block});
  index_.emplace(hash, recency_.begin());
  stats_.insertions += 1;

  while (index_.size() > capacity_) {
    const Entry victim = recency_.back();
    recency_.pop_back();
    index_.erase(victim.hash);
    pool.release_cached_block(victim.block);
    stats_.evictions += 1;
  }
}

std::size_t PrefixCache::evict_until_free(KvBlockPool &pool, std::size_t needed) {
  std::size_t released = 0;
  auto entry = recency_.end();
  while (pool.stats().free_blocks < needed && !recency_.empty()) {
    entry = std::prev(recency_.end());
    const Entry victim = *entry;
    // Refcount above the cache's own single reference means a live sequence is
    // still reading this block. Dropping the cache entry is fine; forcing the
    // block free is not, and the counter records that we declined.
    const bool in_use = pool.block_refcount(victim.block) > 1;
    recency_.pop_back();
    index_.erase(victim.hash);
    pool.release_cached_block(victim.block);
    stats_.evictions += 1;
    if (in_use) {
      stats_.evictions_refused += 1;
    } else {
      released += 1;
    }
  }
  return released;
}

bool PrefixCache::validate(const KvBlockPool &pool, std::string *detail) const {
  if (index_.size() != recency_.size()) {
    if (detail) {
      *detail = "index holds " + std::to_string(index_.size())
        + " entries but the recency list holds " + std::to_string(recency_.size());
    }
    return false;
  }
  for (const Entry &entry : recency_) {
    if (index_.find(entry.hash) == index_.end()) {
      if (detail) *detail = "recency entry missing from the index";
      return false;
    }
    // Every listed block must still be alive: the cache holds a reference, so a
    // refcount of zero means something released it behind the cache's back.
    if (pool.block_refcount(entry.block) == 0) {
      if (detail) {
        *detail = "cached block " + std::to_string(entry.block) + " has refcount 0";
      }
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------- selftest

namespace {

int g_failures = 0;

void check(bool condition, const char *name, const std::string &detail = {}) {
  if (condition) {
    std::printf("  ok   %s\n", name);
    return;
  }
  g_failures += 1;
  std::printf("  FAIL %s%s%s\n", name, detail.empty() ? "" : " -- ", detail.c_str());
}

std::vector<std::uint64_t> token_range(std::uint64_t start, std::size_t count) {
  std::vector<std::uint64_t> tokens(count);
  for (std::size_t index = 0; index < count; index += 1) {
    tokens[index] = start + index;
  }
  return tokens;
}

}  // namespace

int prefix_cache_selftest() {
  g_failures = 0;
  std::printf("prefix_cache selftest\n");
  constexpr std::size_t block_tokens = 16;

  // 1. The chain property. Identical prefixes hash identically; a difference
  //    anywhere earlier must change every later value, which is what lets the
  //    matcher stop at the first miss.
  {
    const auto tokens = token_range(100, 64);
    PrefixHash chain_a = 0;
    PrefixHash chain_b = 0;
    for (std::size_t block = 0; block < 4; block += 1) {
      chain_a = hash_prefix_block(chain_a, tokens.data() + (block * block_tokens), block_tokens);
      chain_b = hash_prefix_block(chain_b, tokens.data() + (block * block_tokens), block_tokens);
    }
    check(chain_a == chain_b, "identical prefixes hash identically");

    auto altered = tokens;
    altered[0] = 999;  // change only the FIRST token
    PrefixHash chain_c = 0;
    for (std::size_t block = 0; block < 4; block += 1) {
      chain_c = hash_prefix_block(chain_c, altered.data() + (block * block_tokens), block_tokens);
    }
    check(chain_c != chain_a, "a change in block 0 changes the final chain value");

    // The same block contents at a different chain position must not alias.
    const PrefixHash standalone = hash_prefix_block(0, tokens.data() + block_tokens, block_tokens);
    const PrefixHash chained = hash_prefix_block(
      hash_prefix_block(0, tokens.data(), block_tokens),
      tokens.data() + block_tokens, block_tokens);
    check(standalone != chained, "the same block hashes differently at a different position");
  }

  // 2. Matching. A shared head is found; a diverging tail is not.
  {
    KvBlockPoolConfig config;
    config.block_tokens = block_tokens;
    config.row_bytes = 4;
    config.block_count = 32;
    KvBlockPool pool(config);
    PrefixCache cache;

    const auto shared = token_range(1, 64);
    PrefixHash chain = 0;
    for (std::size_t block = 0; block < 4; block += 1) {
      chain = hash_prefix_block(chain, shared.data() + (block * block_tokens), block_tokens);
      const KvBlockId id = pool.allocate_for_test();
      cache.insert(chain, id, pool);
    }

    check(cache.match_length_blocks(shared, block_tokens) == 4,
      "a fully cached prompt matches every block");

    // Same first two blocks, then divergence.
    auto partial = shared;
    for (std::size_t index = 2 * block_tokens; index < partial.size(); index += 1) {
      partial[index] = 5000 + index;
    }
    check(cache.match_length_blocks(partial, block_tokens) == 2,
      "matching stops where the prompts diverge");

    // Nothing in common.
    const auto unrelated = token_range(90000, 64);
    check(cache.match_length_blocks(unrelated, block_tokens) == 0,
      "an unrelated prompt matches nothing");

    // A prompt shorter than one block has no whole block to share.
    const auto tiny = token_range(1, block_tokens - 1);
    check(cache.match_length_blocks(tiny, block_tokens) == 0,
      "a sub-block prompt matches nothing");

    std::string detail;
    check(cache.validate(pool, &detail), "cache and pool agree after inserts", detail);
  }

  // 3. Eviction never frees a block a sequence is still reading. This is the
  //    case that would otherwise hand one session another's context.
  {
    KvBlockPoolConfig config;
    config.block_tokens = block_tokens;
    config.row_bytes = 4;
    config.block_count = 8;
    KvBlockPool pool(config);
    PrefixCache cache;

    // One cached block is also held by a live sequence.
    KvSequence live;
    std::vector<std::uint8_t> row(4, 7);
    for (std::size_t position = 0; position < block_tokens; position += 1) {
      pool.append_row(live, row.data(), row.size());
    }
    const KvBlockId busy = live.block_ids().front();
    cache.insert(0xAAAA, busy, pool);

    // And one that nobody else holds. Inserting into the cache HANDS THE BLOCK
    // OVER: the creator drops its own reference afterwards, which is what a
    // finished sequence does when it releases its blocks and leaves the cache
    // keeping them alive. Without that release the block would look in-use to
    // eviction forever, which is how a prefix cache silently stops reclaiming.
    const KvBlockId idle = pool.allocate_for_test();
    cache.insert(0xBBBB, idle, pool);
    pool.release_cached_block(idle);

    const std::size_t before = pool.stats().free_blocks;
    cache.evict_until_free(pool, pool.config().block_count);
    const std::size_t after = pool.stats().free_blocks;

    check(after > before, "eviction frees the blocks it can",
      "free " + std::to_string(before) + " -> " + std::to_string(after));
    check(pool.block_refcount(busy) > 0,
      "a block held by a live sequence survives eviction");
    check(cache.stats().evictions_refused >= 1,
      "declining to free an in-use block is recorded");

    bool intact = true;
    for (std::size_t position = 0; position < block_tokens && intact; position += 1) {
      const std::uint8_t *data = pool.row(live, position);
      intact = data != nullptr && data[0] == 7;
    }
    check(intact, "the live sequence still reads its own bytes after eviction");

    std::string detail;
    check(pool.validate(&detail), "pool accounting valid after eviction", detail);
  }

  // 4. Capacity. An over-full cache evicts oldest-first and stays consistent.
  {
    KvBlockPoolConfig config;
    config.block_tokens = block_tokens;
    config.row_bytes = 4;
    config.block_count = 64;
    KvBlockPool pool(config);
    PrefixCache cache(4);

    for (std::uint64_t entry = 0; entry < 10; entry += 1) {
      const KvBlockId id = pool.allocate_for_test();
      if (id == k_invalid_kv_block) break;
      cache.insert(0x1000 + entry, id, pool);
    }
    check(cache.stats().entries <= 4, "the cache respects its capacity",
      "entries " + std::to_string(cache.stats().entries));
    check(cache.lookup(0x1000) == k_invalid_kv_block, "the oldest entry was evicted first");

    std::string detail;
    check(cache.validate(pool, &detail), "cache valid after capacity eviction", detail);
    check(pool.validate(&detail), "pool valid after capacity eviction", detail);
  }

  std::printf("prefix_cache selftest: %s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}

}  // namespace lizard_native
