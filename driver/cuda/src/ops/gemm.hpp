#pragma once

// ops/: attention, GEMM, MoE, and SSM kernel wrappers shared by every model forward.
//
// Thin cuBLAS wrapper for bf16 matmul.
//
// The transformer linear layers are all of shape `out = act @ W^T`, where
// `act` is row-major [M, K] and `W` is row-major [N, K] (the HF
// safetensors convention). We compute that in cuBLAS's column-major view by
// asking for `W * act^T`, so the kernel sees `(W : K x N) @ (act : M x K)^T`
// → output column-major [N, M] which is the same memory as row-major [M, N].

#include <cstddef>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <memory>

#include "ops/quant_meta.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver::ops {

struct RuntimeQuantScratchSpec {
    std::size_t max_tokens = 0;
    std::size_t max_weight_rows = 0;  // GEMM N
    std::size_t max_weight_cols = 0;  // GEMM K
    std::size_t max_dequant_weight_elems = 0;
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

class RuntimeQuantContext {
public:
    RuntimeQuantContext();
    ~RuntimeQuantContext();
    RuntimeQuantContext(const RuntimeQuantContext&) = delete;
    RuntimeQuantContext& operator=(const RuntimeQuantContext&) = delete;

    void reset() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class ScopedRuntimeQuantContext;
};

class ScopedRuntimeQuantContext {
public:
    explicit ScopedRuntimeQuantContext(RuntimeQuantContext& context) noexcept;
    ~ScopedRuntimeQuantContext();
    ScopedRuntimeQuantContext(const ScopedRuntimeQuantContext&) = delete;
    ScopedRuntimeQuantContext& operator=(const ScopedRuntimeQuantContext&) = delete;

private:
    void* previous_ = nullptr;
};

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
// Dispatcher entry point. Routes supported dense and quantized combinations
// through cuBLAS/cuBLASLt or Marlin, and throws on unsupported combinations
// rather than silently miscomputing.
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

// Streams the LM head GEMM one vocabulary slab at a time, reducing each slab
// to a running argmax before the next overwrites it, so the [M, N] logits
// tensor is never materialised.
//
// The logits round-trip that this removes is pure HBM traffic: at M=512 and a
// 151936 vocabulary the sampler's input is 156 MB written and 156 MB read back
// solely because the reduction was a separate kernel. Slabs stay resident in
// L2, so both ends of that round-trip disappear (§20.36 measured 185 us/step;
// the weight read is untouched and still dominates).
//
// `W` is [N, K] row-major, so a vocabulary slab is a contiguous row range and
// no gather is needed. Results are bit-identical to a full GEMM followed by
// `launch_argmax_bf16`: the tie-break is a total order on (value, -index), so
// where the vocabulary is cut cannot change the answer.
//
//   act      : [M, K] bf16
//   w        : [N, K] bf16 (dense only)
//   token_ids: [M] i32, the output
//   slab     : [M, chunk] bf16 scratch, overwritten every iteration
//   acc_val / acc_idx : [M, kernels::kArgmaxAccumSlots] scratch
//
// Returns false without launching anything when the weight is not dense BF16.
// Callers must not treat that as a fallback: by the time the forward runs, the
// graph key and the epilogue's token source are already committed to the fused
// shape, so the only correct response is to fail loudly. Ask
// `lm_head_argmax_supported` beforehand instead -- this return is the
// belt-and-braces half of the same predicate.
// `chunk` is a device+model property and belongs to the planner's calibration,
// not to a constant in here.
bool lm_head_argmax_chunked(
    cublasHandle_t handle,
    const void* act,
    WeightView w,
    std::int32_t* token_ids,
    void* slab,
    float* acc_val,
    std::int32_t* acc_idx,
    int M, int N, int K,
    int chunk);

// Scratch for `lm_head_argmax_chunked`.
//
// Only the slab is carved from the caller's block, and it is carved from
// `ws.logits` -- by construction the buffer this path is not filling. The size
// is `M * min(chunk, N)` bf16 elements with no padding, and `ws.logits` holds
// `workspace_logits_rows * N` of them, so with `M <= workspace_logits_rows` it
// fits by construction rather than by a runtime test that could quietly fail.
//
// The running accumulators do NOT live here: they are `[rows, kArgmaxAccumSlots]`
// and would push the total past `ws.logits` exactly when every row samples.
// They get their own (tiny) workspace tensors so that "does it fit" is never a
// question the forward has to answer.
std::size_t lm_head_argmax_slab_bytes(int M, int N, int chunk);

// Whether `lm_head_argmax_chunked` can run against this weight. Split out so
// the driver can decide to fuse BEFORE the forward runs -- by then the graph
// key and the epilogue's token source are already committed, and a fallback
// discovered inside the forward would be a silent miscompute.
//
// Note that today's callers all reach this through the implicit
// `WeightView(const DeviceTensor&)`, which hard-codes `scale_data = nullptr`,
// so in practice only the dtype discriminates. The scale check is here because
// the chunked path slices `w.data` by row and has nowhere to carry a scale, so
// a future quantised LM head must not silently qualify.
bool lm_head_argmax_supported(WeightView w);

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

// Grouped variant for sparse-MoE expert buckets. Each group has one GEMM
// with a shared output width `N` and reduction dim `K`, but its own row count
// `M_array[group]`.
//
// POINTER-ARRAY RESIDENCY (measured hazard): cublasGemmGroupedBatchedEx
// does not consume the act/W/y pointer arrays synchronously at call time.
// Transient host arrays handed to back-to-back grouped calls produced
// illegal-address / misaligned-address faults (repro: lora grouped lanes;
// a stream sync between calls or device-resident arrays both cure it).
// Callers must pass DEVICE-RESIDENT pointer arrays whose slots are not
// rewritten while a call may still read them — the nemotron_h MoE and the
// llama_like lora grouping both stage through per-use device slots.
// `M_array` and the internal int/scalar arrays are consumed at call time
// and may be transient host memory.
void gemm_grouped_act_x_wt_bf16(
    cublasHandle_t handle,
    const void* const* act_ptrs_host,
    const void* const* W_ptrs_host,
    void* const*       y_ptrs_host,
    const int*         M_array_host,
    int group_count,
    int N,
    int K,
    float beta = 0.f);

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

// Same storage convention as `gemm_act_x_wt_bf16`, but materialises the
// output as fp32. Used by routers that are specified to compute logits in
// fp32 before top-k selection.
void gemm_act_x_wt_bf16_out_fp32(
    cublasHandle_t handle,
    const void* act,
    const void* W,
    float* y,
    int M,
    int N,
    int K);

// Dense bf16 linear with a broadcast row bias: y[m][n] = sum_k act[m][k] *
// W[n][k] + bias[n]. `bias` may be null, in which case this is exactly
// `gemm_act_x_wt_bf16`.
//
// At M=1 -- the decode shape -- the bias is folded into the GEMV epilogue,
// which removes an entire kernel launch per biased projection. gpt-oss-20b
// has five of them per layer, so at 24 layers that was 120 launches per
// decode step at ~3.6 us each against a 2.2 us empty-launch floor: 11.9% of
// decode GPU time spent re-reading and re-writing a few KB. The fold is
// bit-identical (see `launch_gemv_bf16`).
//
// The fold only happens when the dense autotuner has *already* chosen the
// GEMV for this shape, so a shape where cuBLAS wins is never forced onto a
// slower kernel just to save a launch. Set PIE_GEMV_FUSED_BIAS=0 to disable.
void gemm_act_x_wt_bias_bf16(
    cublasHandle_t handle,
    const void* act, const void* W, const void* bias, void* y,
    int M, int N, int K,
    cudaStream_t stream,
    // `beta = 1` accumulates into `y`, which is what a projection writing
    // straight into a residual wants. The GEMV epilogue folds it next to the
    // bias, so asking for both costs no more launches than asking for either.
    float beta = 0.f);

// Same math as `gemm_act_x_wt_bf16`, but bypasses the cuBLASLt BF16
// dispatcher. This is useful for a few skinny-M packed projections where
// Lt's heuristic is slower than cuBLAS GEMMEx.
void gemm_act_x_wt_bf16_cublas(
    cublasHandle_t handle,
    const void* act, const void* W, void* y,
    int M, int N, int K, float beta = 0.f);

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

// One-shot benchmark of cuBLASLt heuristics + GEMMEx for a given shape.
// Prints per-algo latencies to stderr. Enabled by PIE_BENCH_LM_HEAD_ALGOS=1.
// Runs at most once per process. Safe to call on every MTP step — the second
// and subsequent calls are no-ops.
void maybe_bench_lm_head_algos(
    cublasHandle_t handle,
    const void* act, const void* W, void* y,
    int M, int N, int K);

// ── MLA weight absorption ────────────────────────────────────────────────
// `kv_b_proj` is [heads, qk_nope_dim + v_head_dim, kv_lora_rank] BF16.
//
//   q_latent[n, h, l] = sum_d q_nope[n, h, d] * kv_b[h, d, l]
//   attn_v  [n, h, v] = sum_l latent[n, h, l] * kv_b[h, qk_nope_dim + v, l]
//
// Both are per-head GEMMs over a strided batch, so they run as a single
// `cublasGemmStridedBatchedEx` instead of the scalar one-thread-per-output
// kernels they replace (those stride the inner loop by `kv_lora_rank` and
// reach a small fraction of HBM bandwidth).
void mla_absorb_q_to_latent_bf16(
    cublasHandle_t handle,
    const void* q_nope, const void* kv_b_proj, void* q_latent,
    int tokens, int heads, int qk_nope_dim, int v_head_dim, int kv_lora_rank);

void mla_absorb_latent_to_v_bf16(
    cublasHandle_t handle,
    const void* attn_latent, const void* kv_b_proj, void* attn_v,
    int tokens, int heads, int qk_nope_dim, int v_head_dim, int kv_lora_rank);

}  // namespace pie_cuda_driver::ops
