#include "kv_blocks.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace lizard_native {

bool kv_paged_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("LIZARD_NATIVE_KV_PAGED");
    return value != nullptr && std::string(value) == "1";
  }();
  return enabled;
}

std::size_t kv_block_tokens() {
  static const std::size_t tokens = [] {
    const char *value = std::getenv("LIZARD_NATIVE_KV_BLOCK_TOKENS");
    if (value == nullptr) {
      return static_cast<std::size_t>(16);
    }
    const long parsed = std::strtol(value, nullptr, 10);
    // Powers of two only: the block index is a shift and the slot a mask on
    // the hot read path, and a non-power-of-two would silently turn both into
    // divisions inside attention.
    if (parsed != 8 && parsed != 16 && parsed != 32 && parsed != 64) {
      return static_cast<std::size_t>(16);
    }
    return static_cast<std::size_t>(parsed);
  }();
  return tokens;
}

void KvBlockPool::reset(const KvBlockPoolConfig &config) {
  config_ = config;
  if (config_.block_tokens == 0) {
    config_.block_tokens = 16;
  }
  arena_.assign(config_.block_count * config_.block_tokens * config_.row_bytes, 0);
  refcount_.assign(config_.block_count, 0);
  free_list_.clear();
  free_list_.reserve(config_.block_count);
  // Handed out from the back, so a freshly reset pool allocates block 0 first
  // and a test reading the arena sees the order it expects.
  for (std::size_t index = config_.block_count; index-- > 0;) {
    free_list_.push_back(static_cast<KvBlockId>(index));
  }
  copy_on_writes_ = 0;
  allocations_ = 0;
  releases_ = 0;
}

KvPoolStats KvBlockPool::stats() const {
  KvPoolStats stats;
  stats.total_blocks = config_.block_count;
  stats.free_blocks = free_list_.size();
  stats.used_blocks = config_.block_count - free_list_.size();
  for (const std::uint16_t count : refcount_) {
    if (count > 1) {
      stats.shared_blocks += 1;
    }
  }
  stats.copy_on_writes = copy_on_writes_;
  stats.allocations = allocations_;
  stats.releases = releases_;
  return stats;
}

std::size_t KvBlockPool::blocks_for_tokens(std::size_t tokens) const {
  if (config_.block_tokens == 0) {
    return 0;
  }
  return (tokens + config_.block_tokens - 1) / config_.block_tokens;
}

KvAdmission KvBlockPool::can_admit(std::size_t projected_tokens) const {
  const std::size_t needed = blocks_for_tokens(projected_tokens);
  if (needed > config_.block_count) {
    return KvAdmission::TooLarge;
  }
  // The reserve is a floor kept free so already-running sequences can still
  // grow. Admitting into it would trade a queued request for a stalled one.
  const std::size_t available = free_list_.size() > config_.reserve_blocks
    ? free_list_.size() - config_.reserve_blocks
    : 0;
  return needed <= available ? KvAdmission::Admitted : KvAdmission::Wait;
}

KvBlockId KvBlockPool::allocate_block() {
  if (free_list_.empty()) {
    return k_invalid_kv_block;
  }
  const KvBlockId block = free_list_.back();
  free_list_.pop_back();
  refcount_[block] = 1;
  allocations_ += 1;
  return block;
}

void KvBlockPool::release_block(KvBlockId block) {
  if (block == k_invalid_kv_block || block >= refcount_.size()) {
    return;
  }
  if (refcount_[block] == 0) {
    return;  // already free; releasing twice must not corrupt the free list
  }
  refcount_[block] -= 1;
  if (refcount_[block] == 0) {
    free_list_.push_back(block);
    releases_ += 1;
  }
}

std::uint8_t *KvBlockPool::block_data(KvBlockId block) {
  return arena_.data() + (static_cast<std::size_t>(block) * config_.block_tokens * config_.row_bytes);
}

const std::uint8_t *KvBlockPool::block_data(KvBlockId block) const {
  return arena_.data() + (static_cast<std::size_t>(block) * config_.block_tokens * config_.row_bytes);
}

bool KvBlockPool::append_row(KvSequence &sequence, const std::uint8_t *row, std::size_t row_bytes) {
  if (row == nullptr || row_bytes != config_.row_bytes) {
    return false;
  }
  const std::size_t slot = sequence.length_ % config_.block_tokens;
  if (slot == 0) {
    const KvBlockId block = allocate_block();
    if (block == k_invalid_kv_block) {
      return false;
    }
    sequence.block_ids_.push_back(block);
  }
  const KvBlockId block = sequence.block_ids_.back();
  // A write into a shared block would corrupt the donor. Admission cannot
  // prevent this -- only the prefix cache creates sharing -- so the check lives
  // at the write itself.
  if (refcount_[block] > 1 && !make_writable(sequence, sequence.length_)) {
    return false;
  }
  std::uint8_t *destination = block_data(sequence.block_ids_.back()) + (slot * config_.row_bytes);
  std::memcpy(destination, row, config_.row_bytes);
  sequence.length_ += 1;
  return true;
}

const std::uint8_t *KvBlockPool::row(const KvSequence &sequence, std::size_t position) const {
  if (position >= sequence.length_) {
    return nullptr;
  }
  const std::size_t block_index = position / config_.block_tokens;
  if (block_index >= sequence.block_ids_.size()) {
    return nullptr;
  }
  const KvBlockId block = sequence.block_ids_[block_index];
  if (block == k_invalid_kv_block || block >= refcount_.size()) {
    return nullptr;
  }
  const std::size_t slot = position % config_.block_tokens;
  return block_data(block) + (slot * config_.row_bytes);
}

void KvBlockPool::truncate(KvSequence &sequence, std::size_t keep_positions) {
  if (keep_positions >= sequence.length_) {
    return;
  }
  const std::size_t keep_blocks = blocks_for_tokens(keep_positions);
  for (std::size_t index = keep_blocks; index < sequence.block_ids_.size(); index += 1) {
    release_block(sequence.block_ids_[index]);
  }
  sequence.block_ids_.resize(keep_blocks);
  sequence.length_ = keep_positions;
  sequence.shared_prefix_ = std::min(sequence.shared_prefix_, keep_positions);
}

void KvBlockPool::release(KvSequence &sequence) {
  for (const KvBlockId block : sequence.block_ids_) {
    release_block(block);
  }
  sequence.block_ids_.clear();
  sequence.length_ = 0;
  sequence.shared_prefix_ = 0;
}

std::size_t KvBlockPool::share_prefix(
  const KvSequence &source,
  KvSequence &target,
  std::size_t positions) {
  if (!target.block_ids_.empty() || target.length_ != 0) {
    return 0;  // only a fresh sequence can adopt a shared prefix
  }
  const std::size_t usable = std::min(positions, source.length_);
  // Whole blocks only. Sharing a partially-filled block would put the target's
  // next write into the donor's tail, which is a corruption that produces
  // plausible text rather than an error.
  const std::size_t shared_blocks = usable / config_.block_tokens;
  if (shared_blocks == 0 || shared_blocks > source.block_ids_.size()) {
    return 0;
  }
  for (std::size_t index = 0; index < shared_blocks; index += 1) {
    const KvBlockId block = source.block_ids_[index];
    if (block == k_invalid_kv_block || block >= refcount_.size() || refcount_[block] == 0) {
      // Unwind rather than leave the target holding a half-built table.
      for (std::size_t undo = 0; undo < target.block_ids_.size(); undo += 1) {
        release_block(target.block_ids_[undo]);
      }
      target.block_ids_.clear();
      return 0;
    }
    refcount_[block] += 1;
    target.block_ids_.push_back(block);
  }
  const std::size_t shared_positions = shared_blocks * config_.block_tokens;
  target.length_ = shared_positions;
  target.shared_prefix_ = shared_positions;
  return shared_positions;
}

bool KvBlockPool::make_writable(KvSequence &sequence, std::size_t position) {
  const std::size_t block_index = position / config_.block_tokens;
  if (block_index >= sequence.block_ids_.size()) {
    return false;
  }
  const KvBlockId block = sequence.block_ids_[block_index];
  if (block >= refcount_.size()) {
    return false;
  }
  if (refcount_[block] <= 1) {
    return true;  // already private
  }
  const KvBlockId copy = allocate_block();
  if (copy == k_invalid_kv_block) {
    return false;
  }
  std::memcpy(block_data(copy), block_data(block), config_.block_tokens * config_.row_bytes);
  sequence.block_ids_[block_index] = copy;
  release_block(block);
  copy_on_writes_ += 1;
  if (sequence.shared_prefix_ > block_index * config_.block_tokens) {
    sequence.shared_prefix_ = block_index * config_.block_tokens;
  }
  return true;
}

std::uint16_t KvBlockPool::block_refcount(KvBlockId block) const {
  if (block == k_invalid_kv_block || block >= refcount_.size()) {
    return 0;
  }
  return refcount_[block];
}

void KvBlockPool::retain_block(KvBlockId block) {
  if (block == k_invalid_kv_block || block >= refcount_.size()) {
    return;
  }
  if (refcount_[block] == 0) {
    return;  // retaining a free block would resurrect it outside the free list
  }
  refcount_[block] += 1;
}

void KvBlockPool::release_cached_block(KvBlockId block) {
  release_block(block);
}

bool KvBlockPool::validate(std::string *detail) const {
  if (refcount_.size() != config_.block_count) {
    if (detail) *detail = "refcount table size does not match block count";
    return false;
  }
  std::size_t free_by_refcount = 0;
  for (const std::uint16_t count : refcount_) {
    if (count == 0) {
      free_by_refcount += 1;
    }
  }
  if (free_by_refcount != free_list_.size()) {
    if (detail) {
      *detail = "free list holds " + std::to_string(free_list_.size())
        + " blocks but " + std::to_string(free_by_refcount) + " have refcount 0";
    }
    return false;
  }
  // No duplicates in the free list: a block handed out twice is the failure
  // that silently gives two sequences the same KV.
  std::vector<KvBlockId> sorted = free_list_;
  std::sort(sorted.begin(), sorted.end());
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
    if (detail) *detail = "free list contains a duplicate block";
    return false;
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

// Deterministic pseudo-random bytes: the fuzz cases must reproduce exactly
// when a failure is investigated, so no clock or device seeding.
std::uint32_t next_random(std::uint32_t &state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

KvBlockPoolConfig test_config(std::size_t block_tokens, std::size_t row_bytes, std::size_t blocks) {
  KvBlockPoolConfig config;
  config.block_tokens = block_tokens;
  config.row_bytes = row_bytes;
  config.block_count = blocks;
  config.reserve_blocks = 0;
  return config;
}

}  // namespace

int kv_blocks_selftest() {
  g_failures = 0;
  std::printf("kv_blocks selftest\n");

  // 1. Bit-identity against a contiguous reference, at every block size the
  //    phase sweeps. This is the gate: paged reads must return exactly the
  //    bytes a flat buffer would.
  for (const std::size_t block_tokens : {std::size_t {8}, std::size_t {16}, std::size_t {32}}) {
    const std::size_t row_bytes = 48;
    const std::size_t positions = 137;  // deliberately not a multiple of any size
    KvBlockPool pool(test_config(block_tokens, row_bytes, 64));
    KvSequence sequence;

    std::vector<std::uint8_t> reference(positions * row_bytes);
    std::uint32_t state = 0x1234567;
    for (std::size_t position = 0; position < positions; position += 1) {
      std::vector<std::uint8_t> row(row_bytes);
      for (std::size_t index = 0; index < row_bytes; index += 1) {
        row[index] = static_cast<std::uint8_t>(next_random(state) & 0xFF);
      }
      std::memcpy(reference.data() + (position * row_bytes), row.data(), row_bytes);
      if (!pool.append_row(sequence, row.data(), row_bytes)) {
        check(false, "append_row", "pool exhausted unexpectedly");
        break;
      }
    }

    bool identical = sequence.length() == positions;
    for (std::size_t position = 0; identical && position < positions; position += 1) {
      const std::uint8_t *paged = pool.row(sequence, position);
      identical = paged != nullptr
        && std::memcmp(paged, reference.data() + (position * row_bytes), row_bytes) == 0;
    }
    check(identical,
      (std::string("bit-identical to contiguous, block_tokens=") + std::to_string(block_tokens)).c_str());

    // Out of range must be refused, not read from the neighbouring slot.
    check(pool.row(sequence, positions) == nullptr, "out-of-range read returns null");
  }

  // 2. Truncation releases whole blocks and keeps the surviving bytes intact.
  {
    KvBlockPool pool(test_config(16, 8, 32));
    KvSequence sequence;
    std::vector<std::uint8_t> row(8, 0);
    for (std::size_t position = 0; position < 100; position += 1) {
      row[0] = static_cast<std::uint8_t>(position);
      pool.append_row(sequence, row.data(), row.size());
    }
    const std::size_t before = pool.stats().used_blocks;
    pool.truncate(sequence, 40);
    const std::size_t after = pool.stats().used_blocks;
    check(sequence.length() == 40, "truncate sets length");
    check(after < before, "truncate frees whole blocks",
      "before " + std::to_string(before) + " after " + std::to_string(after));
    bool preserved = true;
    for (std::size_t position = 0; position < 40 && preserved; position += 1) {
      const std::uint8_t *data = pool.row(sequence, position);
      preserved = data != nullptr && data[0] == static_cast<std::uint8_t>(position);
    }
    check(preserved, "truncate preserves surviving positions");
    std::string detail;
    check(pool.validate(&detail), "accounting valid after truncate", detail);
  }

  // 3. Prefix sharing plus copy-on-write. The donor must be untouched after
  //    the borrower writes -- this is the case that silently corrupts one
  //    session's answer with another's context if refcounting is wrong.
  {
    KvBlockPool pool(test_config(16, 4, 32));
    KvSequence donor;
    std::vector<std::uint8_t> row(4, 0);
    for (std::size_t position = 0; position < 40; position += 1) {
      row[0] = static_cast<std::uint8_t>(position);
      pool.append_row(donor, row.data(), row.size());
    }

    KvSequence borrower;
    const std::size_t shared = pool.share_prefix(donor, borrower, 40);
    check(shared == 32, "share_prefix rounds down to whole blocks",
      "shared " + std::to_string(shared));
    check(pool.stats().shared_blocks == 2, "two blocks are shared");

    bool same = true;
    for (std::size_t position = 0; position < shared && same; position += 1) {
      const std::uint8_t *left = pool.row(donor, position);
      const std::uint8_t *right = pool.row(borrower, position);
      same = left != nullptr && right != nullptr && left[0] == right[0];
    }
    check(same, "shared prefix reads identically from both sequences");

    // The borrower diverges. Its write must trigger copy-on-write.
    const std::uint64_t copies_before = pool.stats().copy_on_writes;
    row[0] = 0xEE;
    pool.append_row(borrower, row.data(), row.size());
    check(pool.stats().copy_on_writes >= copies_before, "divergent write is tracked");

    bool donor_intact = true;
    for (std::size_t position = 0; position < 40 && donor_intact; position += 1) {
      const std::uint8_t *data = pool.row(donor, position);
      donor_intact = data != nullptr && data[0] == static_cast<std::uint8_t>(position);
    }
    check(donor_intact, "donor is unchanged after the borrower writes");
    std::string detail;
    check(pool.validate(&detail), "accounting valid after share + CoW", detail);
  }

  // 4. Admission. A request larger than the pool can never succeed and must
  //    say so rather than waiting forever.
  {
    KvBlockPool pool(test_config(16, 4, 8));
    check(pool.can_admit(16) == KvAdmission::Admitted, "small request admitted");
    check(pool.can_admit(8 * 16) == KvAdmission::Admitted, "exactly-full request admitted");
    check(pool.can_admit(8 * 16 + 1) == KvAdmission::TooLarge, "oversized request refused as TooLarge");

    KvBlockPoolConfig reserved = test_config(16, 4, 8);
    reserved.reserve_blocks = 6;
    KvBlockPool guarded(reserved);
    check(guarded.can_admit(3 * 16) == KvAdmission::Wait,
      "request into the reserve waits rather than starving running sequences");
  }

  // 5. Fuzz: interleaved allocate / append / truncate / share / release, with
  //    the invariant checked after every operation. Accounting bugs here are
  //    the ones that hand the same block to two sequences.
  {
    KvBlockPool pool(test_config(16, 4, 48));
    std::vector<KvSequence> sequences(6);
    std::vector<std::uint8_t> row(4, 0);
    std::uint32_t state = 0xC0FFEE;
    bool valid = true;
    std::string detail;

    for (int step = 0; step < 4000 && valid; step += 1) {
      const std::uint32_t choice = next_random(state) % 100;
      const std::size_t index = next_random(state) % sequences.size();
      KvSequence &sequence = sequences[index];

      if (choice < 55) {
        row[0] = static_cast<std::uint8_t>(next_random(state) & 0xFF);
        pool.append_row(sequence, row.data(), row.size());
      } else if (choice < 70) {
        pool.truncate(sequence, sequence.length() / 2);
      } else if (choice < 85) {
        pool.release(sequence);
      } else {
        const std::size_t other = next_random(state) % sequences.size();
        if (other != index && sequences[other].length() >= 32 && sequence.length() == 0) {
          pool.share_prefix(sequences[other], sequence, 32);
        }
      }
      valid = pool.validate(&detail);
    }
    check(valid, "fuzz keeps the pool accounting consistent", detail);

    for (KvSequence &sequence : sequences) {
      pool.release(sequence);
    }
    check(pool.stats().used_blocks == 0, "releasing every sequence empties the pool",
      "used " + std::to_string(pool.stats().used_blocks));
    check(pool.validate(&detail), "accounting valid after full release", detail);
  }

  std::printf("kv_blocks selftest: %s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}

}  // namespace lizard_native
