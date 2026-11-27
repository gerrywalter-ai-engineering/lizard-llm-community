// Lifetime-based activation allocator with free-block merging and safe
// shape-preserving in-place reuse. Clean-room implementation.

#include "alloc.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cat {
namespace {

struct Block { std::size_t offset; std::size_t bytes; };

std::size_t align_up(std::size_t value, std::size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("activation alignment must be a power of two");
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

void merge_blocks(std::vector<Block>& blocks) {
    std::sort(blocks.begin(), blocks.end(), [](const Block& left, const Block& right) {
        return left.offset < right.offset;
    });
    std::vector<Block> merged;
    for (const auto block : blocks) {
        if (block.bytes == 0) continue;
        if (!merged.empty() && merged.back().offset + merged.back().bytes == block.offset) {
            merged.back().bytes += block.bytes;
        } else {
            merged.push_back(block);
        }
    }
    blocks = std::move(merged);
}

}  // namespace

AllocationPlan plan_activations(const Graph& graph, std::size_t alignment) {
    graph.validate();
    AllocationPlan result;
    std::vector<std::size_t> last_use(graph.tensors().size(), 0);
    std::vector<bool> has_use(graph.tensors().size(), false);
    for (std::size_t index = 0; index < graph.nodes().size(); ++index) {
        for (const auto source : graph.nodes()[index].src) {
            last_use[source] = index;
            has_use[source] = true;
        }
    }
    for (const auto output : graph.outputs()) {
        last_use[output] = graph.nodes().size();
        has_use[output] = true;
    }

    std::vector<Block> free_blocks;
    std::unordered_map<TensorId, Allocation> live;
    std::size_t arena_end = 0;

    const auto release_dead_before = [&](std::size_t node_index) {
        std::vector<TensorId> dead;
        for (const auto& [tensor_id, allocation] : live) {
            if (!has_use[tensor_id] || last_use[tensor_id] < node_index) {
                free_blocks.push_back(Block{allocation.offset, allocation.bytes});
                dead.push_back(tensor_id);
            }
        }
        for (const auto tensor_id : dead) live.erase(tensor_id);
        merge_blocks(free_blocks);
    };

    for (std::size_t index = 0; index < graph.nodes().size(); ++index) {
        release_dead_before(index);
        const auto& node = graph.nodes()[index];
        const auto bytes = align_up(graph.tensor(node.dst).bytes(), alignment);
        result.naive_bytes += bytes;

        std::optional<Allocation> selected;
        for (std::size_t source_index = 0; source_index < node.src.size(); ++source_index) {
            const auto source = node.src[source_index];
            const auto live_source = live.find(source);
            if (op_can_inplace(node.op, source_index)
                && live_source != live.end()
                && last_use[source] == index
                && live_source->second.bytes == bytes) {
                selected = Allocation{live_source->second.offset, bytes, true};
                live.erase(live_source);
                break;
            }
        }

        if (!selected) {
            auto best = free_blocks.end();
            for (auto cursor = free_blocks.begin(); cursor != free_blocks.end(); ++cursor) {
                if (cursor->bytes >= bytes && (best == free_blocks.end() || cursor->bytes < best->bytes)) best = cursor;
            }
            if (best != free_blocks.end()) {
                selected = Allocation{best->offset, bytes, false};
                best->offset += bytes;
                best->bytes -= bytes;
                if (best->bytes == 0) free_blocks.erase(best);
            } else {
                const auto offset = align_up(arena_end, alignment);
                selected = Allocation{offset, bytes, false};
                arena_end = offset + bytes;
            }
        }
        result.tensors[node.dst] = *selected;
        live[node.dst] = *selected;
    }
    result.arena_bytes = arena_end;
    return result;
}

}  // namespace cat
