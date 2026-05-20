#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace pie_cuda_driver {

enum class DType : std::uint8_t;

enum class TensorLayoutKind {
    Dense,
    RowPacked,
    AxisConcatenated,
    Grouped,
    QuantPacked,
    View,
};

enum class TensorOwnershipKind {
    Owned,
    BorrowedView,
    Alias,
    Temporary,
};

enum class TensorParallelKind {
    Replicated,
    Column,
    Row,
    Expert,
    Custom,
};

enum class QuantFormat {
    None,
    RuntimeFp8E4M3,
    RuntimeInt8,
    GptqInt4,
    AwqInt4,
    CompressedFp8E4M3,
    CompressedInt8,
    Mxfp4E2M1E8M0,
};

enum class QuantGranularity {
    None,
    PerTensor,
    PerChannel,
    PerGroup,
};

struct QuantSpec {
    QuantFormat format = QuantFormat::None;
    QuantGranularity granularity = QuantGranularity::None;
    int group_size = 0;
    int channel_axis = 0;
    std::string scale_tensor;
    std::string zero_point_tensor;
};

struct TensorSpec {
    std::string name;
    DType dtype;
    std::vector<std::int64_t> shape;
    TensorLayoutKind layout = TensorLayoutKind::Dense;
    TensorOwnershipKind ownership = TensorOwnershipKind::Owned;
    TensorParallelKind parallel = TensorParallelKind::Replicated;
    QuantSpec quant;
    std::string backing_tensor;
};

struct LoadMemoryPlan {
    std::uint64_t persistent_bytes = 0;
    std::uint64_t max_temporary_bytes = 0;
    std::uint64_t estimated_peak_bytes = 0;
};

enum class LoadOpKind {
    Read,
    Copy,
    Slice,
    Shard,
    RowRangeShard,
    GroupedSliceConcat,
    GroupedSlice,
    Cast,
    Concat,
    AxisConcat,
    View,
    Alias,
    Drop,
    QuantizeRuntime,
    Dequantize,
    Deinterleave,
    RepackLayout,
    StackGroups,
    BindMetadata,
    Materialize,
};

struct TensorSourceRef {
    std::string raw_name;
    std::string view_name;
};

struct RawLoadPayload {
    std::string output_name;
    std::string raw_name;
    int shard_axis = -1;
};

struct RowRangeShardPayload {
    std::string output_name;
    std::string raw_name;
    std::int64_t row_offset = 0;
    std::int64_t rows = 0;
};

struct TensorOpPayload {
    std::string output_name;
    std::string secondary_output_name;
    std::vector<std::string> inputs;
    int shard_axis = -1;
};

struct SlicePayload {
    std::string output_name;
    std::vector<std::string> inputs;
    int slice_axis = -1;
    std::int64_t slice_start = 0;
    std::int64_t slice_length = 0;
    int shard_axis = -1;
};

struct AxisConcatPayload {
    std::string output_name;
    int shard_axis = -1;
    std::vector<TensorSourceRef> sources;
};

struct StackGroupsPayload {
    std::string output_name;
    std::string secondary_output_name;
    std::vector<std::string> inputs;
    std::vector<TensorSourceRef> sources;
};

using LoadOpPayload = std::variant<
    RawLoadPayload,
    RowRangeShardPayload,
    TensorOpPayload,
    SlicePayload,
    AxisConcatPayload,
    StackGroupsPayload>;

struct LoadOp {
    LoadOpKind kind = LoadOpKind::Copy;
    LoadOpPayload payload = RawLoadPayload{};
};

struct LoadPlan {
    std::vector<LoadOp> ops;
    std::unordered_map<std::string, TensorSpec> tensors;
    LoadMemoryPlan memory;
    std::size_t axis_concat_groups = 0;
};

struct MaterializedLoadPlan {
    std::uint64_t loaded_bytes = 0;
    std::size_t axis_concat_groups = 0;
    std::size_t planned_tensor_count = 0;
    std::size_t runtime_quantized_weights = 0;
    std::uint64_t runtime_quant_bytes_before = 0;
    std::uint64_t runtime_quant_bytes_after = 0;
};

const char* load_op_kind_name(LoadOpKind kind) noexcept;
const char* tensor_layout_kind_name(TensorLayoutKind kind) noexcept;
const char* quant_format_name(QuantFormat format) noexcept;

LoadOp make_raw_load_op(
    LoadOpKind kind,
    std::string output_name,
    std::string raw_name,
    int shard_axis = -1);
LoadOp make_row_range_shard_op(
    std::string output_name,
    std::string raw_name,
    std::int64_t row_offset,
    std::int64_t rows);
LoadOp make_tensor_op(
    LoadOpKind kind,
    std::string output_name,
    std::vector<std::string> inputs = {},
    std::string secondary_output_name = {},
    int shard_axis = -1);
LoadOp make_slice_op(
    std::string output_name,
    std::string input,
    int slice_axis,
    std::int64_t slice_start,
    std::int64_t slice_length,
    int shard_axis = -1);
LoadOp make_axis_concat_op(
    std::string output_name,
    int shard_axis,
    std::vector<TensorSourceRef> sources);
LoadOp make_stack_groups_op(
    std::string output_name,
    std::string secondary_output_name,
    std::vector<std::string> inputs,
    std::vector<TensorSourceRef> sources = {});

const std::string& load_op_output(const LoadOp& op);
const std::string& load_op_secondary_output(const LoadOp& op);
const std::string& load_op_raw_name(const LoadOp& op);
const std::vector<std::string>& load_op_inputs(const LoadOp& op);
int load_op_shard_axis(const LoadOp& op);
int load_op_slice_axis(const LoadOp& op);
std::int64_t load_op_row_offset(const LoadOp& op);
std::int64_t load_op_rows(const LoadOp& op);
std::int64_t load_op_slice_start(const LoadOp& op);
std::int64_t load_op_slice_length(const LoadOp& op);
const std::vector<TensorSourceRef>& load_op_sources(const LoadOp& op);
void set_load_op_output(LoadOp& op, std::string output_name);

void validate_load_plan(const LoadPlan& plan);
std::string describe_load_plan(const LoadPlan& plan);
std::string dump_load_plan_json(const LoadPlan& plan);

}  // namespace pie_cuda_driver
