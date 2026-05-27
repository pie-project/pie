#pragma once

#include <cstdint>
#include <string>
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
    int view_axis = -1;
    std::int64_t view_start = 0;
    std::int64_t view_length = 0;
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
    std::size_t h2d_copy_count = 0;
    std::size_t h2d_bulk_copy_count = 0;
    std::size_t h2d_pinned_copy_count = 0;
    std::size_t slab_scatter_count = 0;
    std::size_t slab_scatter_placements = 0;
    std::uint64_t h2d_copy_bytes = 0;
    std::uint64_t h2d_bulk_copy_bytes = 0;
    std::uint64_t h2d_pinned_copy_bytes = 0;
    std::uint64_t slab_scatter_source_bytes = 0;
    std::uint64_t slab_scatter_payload_bytes = 0;
    std::size_t copy_stream_flushes = 0;
    std::size_t max_pending_copies_seen = 0;
    std::size_t h2d_batch_calls = 0;
};

}  // namespace pie_cuda_driver
