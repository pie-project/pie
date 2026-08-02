#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tensor.hpp"

namespace pie_cuda_driver {

/// How the store must treat a tensor's memory when it is erased or replaced.
///
/// The one piece of per-tensor bookkeeping the tensor itself cannot answer:
/// `DeviceTensor::owns_memory()` says whether *this* handle frees the block,
/// but not whether some other entry is still viewing it.
enum class TensorOwnershipKind {
    Owned,
    BorrowedView,
    Alias,
    Temporary,
    /// Memory this store neither allocated nor views through another entry.
    ///
    /// A store built over a caller-supplied arena -- one slot of a streamed
    /// group's slab -- holds tensors whose bytes outlive it and are freed by
    /// whoever lent the arena. `Owned` would be a lie the invariant check
    /// catches, and `BorrowedView` demands a backing tensor that by definition
    /// is not in this store, so neither could describe it.
    External,
};

/// What the store knows about a resident tensor beyond its bytes.
///
/// Deliberately small. Everything here has a reader: `dtype`/`shape` are the
/// plan's claim, checked against the tensor on insert and re-checked by
/// `validate_invariants`; `ownership` and `backing_tensor` drive erase and
/// alias resolution. Fields that only ever round-tripped through the artifact
/// codec (layout, tensor-parallel kind, a second quantization description)
/// were removed — the live quant metadata is `ops::QuantMeta`, which the GEMM
/// path actually reads.
struct TensorDecl {
    std::string name;
    DType dtype = DType::BF16;
    std::vector<std::int64_t> shape;
    TensorOwnershipKind ownership = TensorOwnershipKind::Owned;
    std::string backing_tensor;
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
    std::uint64_t h2d_copy_bytes = 0;
    std::uint64_t h2d_bulk_copy_bytes = 0;
    std::uint64_t h2d_pinned_copy_bytes = 0;
    std::size_t copy_stream_flushes = 0;
    std::size_t max_pending_copies_seen = 0;
    std::size_t h2d_batch_calls = 0;
    // Phase timing (ms). Always accumulated (cost is a few clock reads per
    // instruction); surfaced only when PIE_LOAD_EXECUTOR_PROFILE is set.
    double phase_alloc_ms = 0;          // device buffer/arena allocation
    double phase_transfer_ms = 0;       // staged H2D copy + stream sync (flush)
    double phase_transform_ms = 0;      // tile-map / finalize (GPU)
    double phase_pinned_alloc_ms = 0;   // one-time reader-lane pinned staging alloc
};

}  // namespace pie_cuda_driver
