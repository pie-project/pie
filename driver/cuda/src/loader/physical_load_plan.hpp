#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "loader/load_plan.hpp"
#include "loader/safetensors.hpp"

namespace pie_cuda_driver {

enum class PhysicalTransformKind {
    Cast,
    Dequantize,
};

// A physical checkpoint-to-device copy. The source is a safetensors tensor
// plus optional row-major rectangular slices; the destination is an offset
// inside a planned runtime tensor allocation.
struct ByteRangeWrite {
    std::size_t op_index = 0;
    std::string op_kind;
    std::string raw_name;
    std::string output_name;
    std::vector<TensorSlice> slices;
    std::vector<std::int64_t> dst_shape;
    std::uint64_t dst_offset_bytes = 0;
    std::uint64_t bytes = 0;
    std::uint64_t range_count = 0;
    bool contiguous = true;
};

// A non-copy physical transform scheduled under a bounded tile/scratch budget.
// The first implementation records exact physical requirements and lets the
// materializer execute copy-free byte writes. Tiled execution hooks are kept
// generic so Cast/Dequantize can be fused with raw reads as the executor grows.
struct TiledTransform {
    std::size_t op_index = 0;
    PhysicalTransformKind kind = PhysicalTransformKind::Cast;
    std::string output_name;
    std::vector<std::string> inputs;
    std::uint64_t input_bytes = 0;
    std::uint64_t output_bytes = 0;
    std::uint64_t tile_bytes = 0;
    std::uint64_t scratch_bytes = 0;
};

struct PhysicalLoadMemoryPlan {
    std::uint64_t persistent_bytes = 0;
    std::uint64_t semantic_max_temporary_bytes = 0;
    std::uint64_t max_copy_temporary_bytes = 0;
    std::uint64_t max_transform_scratch_bytes = 0;
    std::uint64_t max_temporary_bytes = 0;
    std::uint64_t estimated_peak_bytes = 0;
    std::uint64_t checkpoint_read_bytes = 0;
    std::uint64_t device_write_bytes = 0;
    std::uint64_t byte_write_count = 0;
    std::uint64_t byte_range_count = 0;
    std::uint64_t tiled_transform_count = 0;
};

struct PhysicalLoadPlan {
    std::vector<ByteRangeWrite> byte_writes;
    std::vector<TiledTransform> tiled_transforms;
    PhysicalLoadMemoryPlan memory;
};

std::vector<ByteRangeWrite> lower_byte_writes_for_op(
    const LoadOp& op,
    std::size_t op_index,
    const LoadPlan& plan,
    const TensorMetadataSource& metadata,
    int tp_rank,
    int tp_size);

PhysicalLoadPlan build_physical_load_plan(
    const LoadPlan& plan,
    const TensorMetadataSource& metadata,
    int tp_rank,
    int tp_size,
    std::uint64_t transform_tile_bytes = 64ull * 1024ull * 1024ull);

const char* physical_transform_kind_name(PhysicalTransformKind kind) noexcept;
std::string describe_physical_load_plan(const PhysicalLoadPlan& plan);
std::string dump_physical_load_plan_json(const PhysicalLoadPlan& plan);
std::string dump_load_plan_json(
    const LoadPlan& plan,
    const PhysicalLoadPlan& physical_plan);

}  // namespace pie_cuda_driver
