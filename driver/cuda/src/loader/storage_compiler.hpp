#pragma once

#include "loader/checkpoint_source.hpp"
#include "loader/layout_plan.hpp"
#include "loader/storage_program.hpp"

namespace pie_cuda_driver {

StorageProgram compile_storage_program(
    const LayoutPlan& plan,
    const CheckpointSource& metadata,
    int tp_rank,
    int tp_size,
    std::uint64_t transform_tile_bytes = 64ull * 1024ull * 1024ull,
    StorageOptimizerConfig optimizer = {});

}  // namespace pie_cuda_driver

