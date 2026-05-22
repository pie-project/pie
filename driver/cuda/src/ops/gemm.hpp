#pragma once

// Thin cuBLAS wrapper for bf16 matmul.
//
// The transformer linear layers are all of shape `out = act @ W^T`, where
// `act` is row-major [M, K] and `W` is row-major [N, K] (the HF
// safetensors convention). We compute that in cuBLAS's column-major view by
// asking for `W * act^T`, so the kernel sees `(W : K x N) @ (act : M x K)^T`
// → output column-major [N, M] which is the same memory as row-major [M, N].

#include <cstddef>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <memory>

#include "model/loaded_model.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver::ops {

struct RuntimeQuantScratchSpec {
    std::size_t max_tokens = 0;
    std::size_t max_weight_rows = 0;  // GEMM N
    std::size_t max_weight_cols = 0;  // GEMM K
    bool has_fp8 = false;
    bool has_int8 = false;

    bool empty() const noexcept {
        return max_tokens == 0 ||
               max_weight_rows == 0 ||
               max_weight_cols == 0 ||
               (!has_fp8 && !has_int8);
    }
};

std::size_t runtime_quant_scratch_bytes(const RuntimeQuantScratchSpec& spec);

// Preallocate the runtime-quant GEMM scratch described by `spec`. When
// `seal_after_reserve` is true, any later attempt to grow those buffers throws
// instead of reallocating, preserving CUDA graph pointer stability.
void reserve_runtime_quant_scratch(
    const RuntimeQuantScratchSpec& spec,
    bool seal_after_reserve);

// Lightweight reference to a weight tensor + (optional) quantization
// metadata, threaded through the GEMM dispatcher. The bf16 path takes the
// implicit `WeightView(const DeviceTensor&)` constructor and pays nothing;
// the quantized path uses `WeightView::quantized(...)`.
//
// We deliberately don't store a `const DeviceTensor*` — bind functions may
// hand us raw pointers (e.g. into a fused expert table) that don't have a
// DeviceTensor wrapper. Carrying just `(data, dtype)` covers both cases.
struct WeightView {
    const void* data = nullptr;
    DType       dtype = DType::BF16;
    std::size_t nbytes = 0;

    // Quant metadata. `scale_data == nullptr` means "no quant — bf16 path".
    // For per-channel / per-group quant the dispatcher reads the layout
    // hints from `kind`, `group_size`, `channel_axis`.
    const void*       scale_data = nullptr;
    DType             scale_dtype = DType::FP32;
    std::size_t       scale_numel = 0;
    QuantMeta::Kind   quant_kind = QuantMeta::Kind::PerTensor;
    const void*       zero_point_data = nullptr;
    int               group_size = 0;
    int               channel_axis = 0;

    WeightView() = default;

    // Implicit conversion from a plain DeviceTensor — preserves call-site
    // terseness for the unquantized path (the 99% case in M0).
    WeightView(const DeviceTensor& t)
        : data(t.data()), dtype(t.dtype()), nbytes(t.nbytes()) {}

    // Raw pointer + dtype, for buffers without a DeviceTensor handle
    // (deinterleaved MoE scratch, expert pointer arrays).
    static WeightView raw(const void* p, DType d) {
        WeightView v; v.data = p; v.dtype = d; return v;
    }

    // Quantized weight: ties together a weight DeviceTensor and a
    // `QuantMeta` snapshot pulled from `LoadedModel::quant_meta`.
    static WeightView quantized(const DeviceTensor& weight, const QuantMeta& meta) {
        WeightView v;
        v.data = weight.data();
        v.dtype = weight.dtype();
        v.nbytes = weight.nbytes();
        v.scale_data = meta.scale ? meta.scale->data() : nullptr;
        v.scale_dtype = meta.scale ? meta.scale->dtype() : DType::FP32;
        v.scale_numel = meta.scale ? meta.scale->numel() : 0;
        v.quant_kind = meta.kind;
        v.zero_point_data = meta.zero_point ? meta.zero_point->data() : nullptr;
        v.group_size = meta.group_size;
        v.channel_axis = meta.channel_axis;
        return v;
    }

    static WeightView mxfp4_marlin(
        const DeviceTensor& weight,
        const DeviceTensor& scale)
    {
        WeightView v;
        v.data = weight.data();
        v.dtype = DType::MXFP4_PACKED;
        v.nbytes = weight.nbytes();
        v.scale_data = scale.data();
        v.scale_dtype = DType::UINT8;
        v.scale_numel = scale.numel();
        v.quant_kind = QuantMeta::Kind::PerGroup;
        v.group_size = 32;
        v.channel_axis = 0;
        return v;
    }
};

class CublasHandle {
public:
    explicit CublasHandle(cudaStream_t stream = 0);
    ~CublasHandle();

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t handle() const noexcept { return h_; }
    void set_stream(cudaStream_t s);
    // Stream currently bound to the cublas handle. Used by per-arch
    // forward bodies so their loose `<<<grid, block, 0, s>>>` kernel
    // launches stay on the same stream as cublas — required for CUDA
    // graph capture to record every kernel.
    cudaStream_t stream() const noexcept;

private:
    cublasHandle_t h_ = nullptr;
};

// Computes y = alpha * act @ W^T + beta * y. All bf16, fp32 accumulation.
//   act: [M, K]
//   W:   [N, K]   (row-major; the HF storage convention)
//   y:   [M, N]
// Default beta = 0 (overwrite). Pass beta = 1 to fuse a residual add.
//
// Dispatcher entry point. Currently routes only `(BF16, BF16)` to the
// existing cuBLAS path; M1 adds `(BF16, FP8_E4M3)` and M3 adds
// `(BF16, INT4_PACKED)` via marlin. Throws on unsupported combos rather
// than silently miscomputing.
//
// `act_dtype` / `y_dtype` default to BF16 so the common-case call site
// just passes `(handle, act, w_tensor, y, M, N, K[, beta])` unchanged
// from the deprecated `gemm_act_x_wt_bf16` shape.
void gemm_act_x_w(
    cublasHandle_t handle,
    const void* act,
    WeightView w,
    void* y,
    int M, int N, int K,
    float beta = 0.f,
    DType act_dtype = DType::BF16,
    DType y_dtype = DType::BF16);

// Batched variant: per-batch act / W / y pointers, all sharing the same
// (M, N, K). Used by the Qwen3.6-MoE decode path to fuse the per-expert
// `gate_up_proj` GEMM (and the `down_proj` GEMM) of all top-K active
// experts into a single cuBLAS launch.
//
//   act_ptrs_dev : [batch] device array of pointers to [M, K] act_dtype
//   w_ptrs_dev   : [batch] device array of pointers to [N, K] w_dtype
//   y_ptrs_dev   : [batch] device array of pointers to [M, N] y_dtype
//
// All batch elements share `(act_dtype, w_dtype, y_dtype)`. M0 supports
// only `(BF16, BF16, BF16)`. M2 will add per-batch scale pointers for
// per-expert MoE FP8.
void gemm_batched_act_x_w(
    cublasHandle_t handle,
    const void* const* act_ptrs_dev,
    const void* const* w_ptrs_dev,
    void* const*       y_ptrs_dev,
    int M, int N, int K,
    int batch_count,
    float beta = 0.f,
    DType act_dtype = DType::BF16,
    DType w_dtype = DType::BF16,
    DType y_dtype = DType::BF16);

// ── Legacy bf16-only entry points ─────────────────────────────────────
// Thin wrappers around the dispatchers above. Kept as the primary entry
// point for archs whose forward functions haven't been migrated to
// `make_weight_view` yet (gemma2/3/4, gpt_oss, mixtral, gemma3n,
// qwen3_5_moe). Functionally equivalent to calling `gemm_act_x_w` with
// a `WeightView::raw(W, DType::BF16)` — bf16-only, no quant. Migrate to
// `gemm_act_x_w` + `make_weight_view` when adding quant support to that
// arch.
inline void gemm_act_x_wt_bf16(
    cublasHandle_t handle,
    const void* act, const void* W, void* y,
    int M, int N, int K, float beta = 0.f)
{
    gemm_act_x_w(handle, act, WeightView::raw(W, DType::BF16),
                 y, M, N, K, beta);
}

inline void gemm_batched_act_x_wt_bf16(
    cublasHandle_t handle,
    const void* const* act_ptrs_dev,
    const void* const* W_ptrs_dev,
    void* const*       y_ptrs_dev,
    int M, int N, int K, int batch_count, float beta = 0.f)
{
    gemm_batched_act_x_w(handle,
                         act_ptrs_dev, W_ptrs_dev, y_ptrs_dev,
                         M, N, K, batch_count, beta);
}

}  // namespace pie_cuda_driver::ops
