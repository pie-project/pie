#include "loader/storage_compiler.hpp"

namespace pie_cuda_driver {

StorageProgram build_storage_program(
    const LayoutPlan& plan,
    const CheckpointSource& metadata,
    int tp_rank,
    int tp_size,
    std::uint64_t transform_tile_bytes,
    StorageOptimizerConfig optimizer);

StorageProgram compile_storage_program(
    const LayoutPlan& plan,
    const CheckpointSource& metadata,
    int tp_rank,
    int tp_size,
    std::uint64_t transform_tile_bytes,
    StorageOptimizerConfig optimizer)
{
    return build_storage_program(
        plan, metadata, tp_rank, tp_size, transform_tile_bytes, optimizer);
}

}  // namespace pie_cuda_driver
