#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "loader/layout_plan.hpp"
#include "loader/safetensors.hpp"

namespace pie_cuda_driver {

inline constexpr std::size_t kInvalidStorageId =
    static_cast<std::size_t>(-1);

enum class TileMapKind {
    Cast,
    Decode,
    Encode,
    Transcode,
    Reblock,
    Reorder,
};

enum class StorageActionKind {
    Allocate,
    ExtentWrite,
    TileMap,
    Transform,
    Attach,
    CreateView,
    Release,
    Finalize,
};

enum class StorageInstrKind {
    Allocate,
    ExtentWrite,
    TileMap,
    Transform,
    Attach,
    CreateView,
    Release,
    Finalize,
};

enum class StorageTransformKind {
    None,
    Slice,
    Concat,
    Quantize,
    Materialize,
    Decode,
    Deinterleave,
    Repack,
    Stack,
};

struct StorageSlicePayload {
    int axis = -1;
    std::int64_t start = 0;
    std::int64_t length = 0;
    int shard_axis = -1;
};

struct StorageViewPayload {
    int axis = -1;
    std::int64_t start = 0;
    std::int64_t length = 0;
};

struct StorageAxisPayload {
    int shard_axis = -1;
};

using StorageInstrPayload = std::variant<
    std::monostate,
    StorageSlicePayload,
    StorageViewPayload,
    StorageAxisPayload>;

// A storage checkpoint-to-device copy. The source is a safetensors tensor
// plus optional row-major rectangular slices; the destination is an offset
// inside a planned runtime tensor allocation.
struct ExtentWrite {
    std::size_t op_index = kInvalidStorageId;
    LayoutExprId expr_id = kInvalidStorageId;
    std::size_t binding_index = kInvalidStorageId;
    std::string op_kind;
    std::string raw_name;
    std::string output_name;
    std::vector<TensorSlice> slices;
    std::vector<std::int64_t> dst_shape;
    std::uint64_t dst_offset_bytes = 0;
    std::string source_path;
    std::uint32_t source_shard_id = 0;
    std::uint64_t source_offset_bytes = 0;
    std::uint64_t source_span_bytes = 0;
    std::uint64_t bytes = 0;
    std::uint64_t range_count = 0;
    bool contiguous = true;
};

// A non-copy storage transform scheduled under a bounded tile/scratch budget.
// The first implementation records exact storage requirements and lets the
// load_executor execute copy-free extent writes. Tiled execution hooks are kept
// generic so Cast/Decode/Encode/Reblock can be fused with raw reads as the
// executor grows.
struct TileMap {
    std::size_t op_index = kInvalidStorageId;
    LayoutExprId expr_id = kInvalidStorageId;
    std::size_t binding_index = kInvalidStorageId;
    TileMapKind kind = TileMapKind::Cast;
    std::string output_name;
    std::vector<std::string> inputs;
    std::uint64_t input_bytes = 0;
    std::uint64_t output_bytes = 0;
    std::uint64_t tile_bytes = 0;
    std::uint64_t scratch_bytes = 0;
};

struct StorageInstr {
    std::size_t step_index = 0;
    StorageInstrKind kind =
        StorageInstrKind::Transform;
    StorageTransformKind transform_kind = StorageTransformKind::None;
    std::size_t op_index = kInvalidStorageId;
    LayoutExprId expr_id = kInvalidStorageId;
    std::size_t binding_index = kInvalidStorageId;
    std::vector<std::size_t> extent_write_indices;
    std::size_t tile_map_index = kInvalidStorageId;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<std::size_t> dependencies;
    StorageInstrPayload payload = std::monostate{};
};

struct StorageMemoryPlan {
    std::uint64_t persistent_bytes = 0;
    std::uint64_t layout_max_temporary_bytes = 0;
    std::uint64_t max_extent_temporary_bytes = 0;
    std::uint64_t max_transform_scratch_bytes = 0;
    std::uint64_t max_temporary_bytes = 0;
    std::uint64_t estimated_peak_bytes = 0;
    std::uint64_t checkpoint_read_bytes = 0;
    std::uint64_t device_write_bytes = 0;
    std::uint64_t extent_write_count = 0;
    std::uint64_t algebra_extent_write_count = 0;
    std::uint64_t extent_range_count = 0;
    std::uint64_t tile_map_count = 0;
    std::uint64_t optimized_extent_write_count = 0;
    std::uint64_t coalesced_extent_write_count = 0;
    std::uint64_t scheduled_step_count = 0;
    std::uint64_t file_ordered_extent_write_count = 0;
};

struct StorageProgram {
    std::vector<ExtentWrite> extent_writes;
    std::vector<TileMap> tile_maps;
    std::vector<StorageInstr> schedule;
    std::vector<std::size_t> scheduled_extent_writes;
    StorageMemoryPlan memory;
};

struct StorageOptimizerConfig {
    bool enabled = true;
    bool coalesce_adjacent = true;
};

const char* storage_action_kind_name(StorageActionKind kind) noexcept;
const char* storage_instr_kind_name(
    StorageInstrKind kind) noexcept;
const char* storage_transform_kind_name(
    StorageTransformKind kind) noexcept;
const char* tile_map_kind_name(TileMapKind kind) noexcept;
void validate_storage_program(
    const LayoutPlan& layout_plan,
    const StorageProgram& storage_program);
std::string describe_storage_program(const StorageProgram& plan);
std::string dump_storage_program_json(const StorageProgram& plan);
std::string dump_layout_plan_json(
    const LayoutPlan& plan,
    const StorageProgram& storage_program);

}  // namespace pie_cuda_driver
