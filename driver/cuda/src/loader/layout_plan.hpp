#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.hpp"

namespace pie_cuda_driver {

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

struct TensorDecl {
    std::string name;
    DType dtype = DType::BF16;
    std::vector<std::int64_t> shape;
    TensorLayoutKind layout = TensorLayoutKind::Dense;
    TensorOwnershipKind ownership = TensorOwnershipKind::Owned;
    TensorParallelKind parallel = TensorParallelKind::Replicated;
    QuantSpec quant;
    std::string backing_tensor;
};

struct LayoutMemoryPlan {
    std::uint64_t persistent_bytes = 0;
    std::uint64_t max_temporary_bytes = 0;
    std::uint64_t estimated_peak_bytes = 0;
};

using LayoutExprId = std::size_t;

enum class LayoutExprKind {
    Source,
    Select,
    Partition,
    Join,
    Stack,
    Unzip,
    Reorder,
    View,
    Cast,
    Encode,
    Decode,
    Transcode,
    Attach,
    Release,
    Realize,
};

struct LayoutExpr {
    LayoutExprKind kind = LayoutExprKind::Source;
    std::vector<LayoutExprId> inputs;
    TensorDecl decl;
    std::string raw_name;
    std::string runtime_name;
    std::string secondary_runtime_name;
    int axis = -1;
    std::int64_t start = 0;
    std::int64_t length = 0;
    int partitions = 0;
    int partition_index = 0;
    DType dtype;
    QuantSpec encoding;
};

struct LayoutBinding {
    std::string runtime_name;
    LayoutExprId root = 0;
};

struct LayoutAlgebra {
    std::vector<LayoutExpr> exprs;
    std::vector<LayoutBinding> bindings;
};

struct TensorSourceRef {
    std::string raw_name;
    std::string view_name;
};

struct LayoutPlan {
    LayoutAlgebra algebra;
    std::unordered_map<std::string, TensorDecl> tensors;
    LayoutMemoryPlan memory;
    std::size_t axis_concat_groups = 0;
};

struct LoadExecutionStats {
    std::uint64_t loaded_bytes = 0;
    std::size_t axis_concat_groups = 0;
    std::size_t planned_tensor_count = 0;
    std::size_t runtime_quantized_weights = 0;
    std::uint64_t runtime_quant_bytes_before = 0;
    std::uint64_t runtime_quant_bytes_after = 0;
    std::uint64_t planned_storage_peak_bytes = 0;
    std::uint64_t planned_storage_temp_bytes = 0;
    std::uint64_t cuda_total_bytes = 0;
    std::uint64_t cuda_free_before_bytes = 0;
    std::uint64_t cuda_min_free_bytes = 0;
    std::uint64_t cuda_free_after_bytes = 0;
    std::uint64_t cuda_actual_peak_delta_bytes = 0;
    std::size_t cuda_memory_samples = 0;
};

const char* layout_expr_kind_name(LayoutExprKind kind) noexcept;
const char* tensor_layout_kind_name(TensorLayoutKind kind) noexcept;
const char* quant_format_name(QuantFormat format) noexcept;

void validate_layout_plan(const LayoutPlan& plan);
std::string describe_layout_plan(const LayoutPlan& plan);
std::string dump_layout_plan_json(const LayoutPlan& plan);

}  // namespace pie_cuda_driver
