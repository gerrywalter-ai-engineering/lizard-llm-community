#pragma once

#include "graph.h"

#include <cstddef>
#include <unordered_map>

namespace cat {

struct Allocation {
    std::size_t offset{};
    std::size_t bytes{};
    bool inplace{false};
};

struct AllocationPlan {
    std::unordered_map<TensorId, Allocation> tensors;
    std::size_t arena_bytes{};
    std::size_t naive_bytes{};
};

[[nodiscard]] AllocationPlan plan_activations(const Graph& graph, std::size_t alignment = 64);

}  // namespace cat
