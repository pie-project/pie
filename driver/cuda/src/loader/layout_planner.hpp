#pragma once

#include "loader/checkpoint_source.hpp"
#include "loader/layout_plan.hpp"
#include "loader/runtime_abi.hpp"
#include "loader/semantic_graph.hpp"

namespace pie_cuda_driver {

// Native semantic-to-algebra planner. This is the migration target for
// model_schema.cpp: adapters declare SemanticGraph groups, RuntimeABI declares
// final contracts, and this planner emits LayoutAlgebra directly.
class LayoutPlanner {
public:
    explicit LayoutPlanner(const RuntimeABI& runtime_abi) noexcept;

    LayoutPlan build_dense_algebra_plan(
        const SemanticGraph& graph,
        const CheckpointSource& source,
        int tp_size) const;

private:
    const RuntimeABI& runtime_abi_;
};

LayoutPlan build_native_dense_algebra_plan(
    const SemanticGraph& graph,
    const CheckpointSource& source,
    int tp_size,
    const RuntimeABI& runtime_abi = pie_cuda_runtime_abi());

}  // namespace pie_cuda_driver
