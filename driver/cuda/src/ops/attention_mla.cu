#include "ops/attention_mla.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>

#include <cuda_bf16.h>
#include <math_constants.h>

#include <flashinfer/attention/mla.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/fastdiv.cuh>

#include "cuda_check.hpp"
#include "ops/attention_mla_naive.cuh"

namespace pie_cuda_driver::ops {
namespace {

using DTypeQ = __nv_bfloat16;
using DTypeKV = __nv_bfloat16;
using DTypeO = __nv_bfloat16;
using IdType = int32_t;

template <typename T>
inline T* offset_ptr(void* base, std::int64_t off) {
    return reinterpret_cast<T*>(reinterpret_cast<std::uint8_t*>(base) + off);
}

}  // namespace

struct MlaPlanCache {
    ::flashinfer::MLAPlanInfo plan_info;
    int total_tokens = 0;
    int num_requests = 0;
    int num_heads = 0;
    int kv_lora_rank = 0;
    int qk_rope_head_dim = 0;
    int page_size = 0;
    bool causal = false;
    float sm_scale = 1.f;
    bool valid = false;
    std::vector<IdType> qo_h_buf;
    std::vector<IdType> kv_h_buf;
    std::vector<IdType> kv_len_h_buf;
};

void MlaPlanCacheDeleter::operator()(MlaPlanCache* p) const noexcept {
    delete p;
}

MlaPlanCachePtr make_mla_plan() {
    return MlaPlanCachePtr(new MlaPlanCache{});
}

namespace {

bool mla_use_naive_backend() {
    static const int choice = [] {
        int dev = 0;
        cudaGetDevice(&dev);
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        return major >= 10 ? 1 : 0;
    }();
    return choice == 1;
}

}  // namespace

void plan_attention_mla_bf16(
    MlaPlanCache& cache,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    const std::uint32_t* kv_last_page_lens_h,
    int total_tokens,
    int num_requests,
    int num_heads,
    int kv_lora_rank,
    int qk_rope_head_dim,
    int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    bool causal,
    float sm_scale)
{
    if (kv_lora_rank != 512 || qk_rope_head_dim != 64) {
        throw std::runtime_error(
            "flashinfer MLA: this build currently instantiates only "
            "kv_lora_rank=512, qk_rope_head_dim=64");
    }
    // The naive/mma backend reads the geometry straight off the device
    // indptr arrays and never touches `plan_info`, so on that path the whole
    // FlashInfer scheduler is dead work — and it is not cheap: `MLAPlan`
    // measured 2.5ms of host CPU per decode step at 128 requests (`enq.attn_plan`
    // under PIE_STEP_PROFILE), which is longer than the entire GPU wave and
    // was the single largest host item in the engine.
    if (!mla_use_naive_backend()) {
        cache.qo_h_buf.resize(num_requests + 1);
        cache.kv_h_buf.resize(num_requests + 1);
        cache.kv_len_h_buf.resize(num_requests);
        for (int r = 0; r <= num_requests; ++r) {
            cache.qo_h_buf[r] = static_cast<IdType>(qo_indptr_h[r]);
            cache.kv_h_buf[r] = static_cast<IdType>(kv_page_indptr_h[r]);
        }
        for (int r = 0; r < num_requests; ++r) {
            const int pages = static_cast<int>(kv_page_indptr_h[r + 1] -
                                               kv_page_indptr_h[r]);
            cache.kv_len_h_buf[r] =
                static_cast<IdType>((pages - 1) * page_size +
                                    static_cast<int>(kv_last_page_lens_h[r]));
        }

        CUDA_CHECK(::flashinfer::MLAPlan<IdType>(
            workspace.float_buffer(), workspace.float_bytes(),
            workspace.int_buffer(), workspace.page_locked_int(),
            workspace.int_bytes(),
            cache.plan_info,
            cache.qo_h_buf.data(),
            cache.kv_h_buf.data(),
            cache.kv_len_h_buf.data(),
            static_cast<std::uint32_t>(num_requests),
            static_cast<std::uint32_t>(num_heads),
            static_cast<std::uint32_t>(kv_lora_rank),
            causal,
            stream));
    }

    cache.total_tokens = total_tokens;
    cache.num_requests = num_requests;
    cache.num_heads = num_heads;
    cache.kv_lora_rank = kv_lora_rank;
    cache.qk_rope_head_dim = qk_rope_head_dim;
    cache.page_size = page_size;
    cache.causal = causal;
    cache.sm_scale = sm_scale > 0.f
        ? sm_scale
        : 1.0f / std::sqrt(static_cast<float>(kv_lora_rank + qk_rope_head_dim));
    cache.valid = true;
}

namespace {

template <::flashinfer::MaskMode MASK>
void dispatch_mla_512_64(
    const MlaPlanCache& cache,
    const void* q_nope,
    const void* q_pe,
    MlaCacheLayerView layer,
    void* o,
    const std::uint32_t* kv_page_indices_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    float* lse_out)
{
    using Params = ::flashinfer::MLAParams<DTypeQ, DTypeKV, DTypeO, IdType>;
    Params params;
    void* int_buf = workspace.int_buffer();
    void* float_buf = workspace.float_buffer();
    const auto& p = cache.plan_info;

    params.q_nope = const_cast<DTypeQ*>(static_cast<const DTypeQ*>(q_nope));
    params.q_pe = const_cast<DTypeQ*>(static_cast<const DTypeQ*>(q_pe));
    params.ckv = static_cast<DTypeKV*>(layer.ckv_pages);
    params.kpe = static_cast<DTypeKV*>(layer.kpe_pages);
    params.final_o = static_cast<DTypeO*>(o);
    params.final_lse = lse_out;
    params.partial_o = offset_ptr<DTypeO>(float_buf, p.partial_o_offset);
    params.partial_lse = offset_ptr<float>(float_buf, p.partial_lse_offset);

    params.q_indptr = offset_ptr<IdType>(int_buf, p.q_indptr_offset);
    params.kv_indptr = offset_ptr<IdType>(int_buf, p.kv_indptr_offset);
    params.partial_indptr = offset_ptr<IdType>(int_buf, p.partial_indptr_offset);
    params.merge_packed_offset_start =
        offset_ptr<IdType>(int_buf, p.merge_packed_offset_start_offset);
    params.merge_packed_offset_end =
        offset_ptr<IdType>(int_buf, p.merge_packed_offset_end_offset);
    params.merge_partial_packed_offset_start =
        offset_ptr<IdType>(int_buf, p.merge_partial_packed_offset_start_offset);
    params.merge_partial_packed_offset_end =
        offset_ptr<IdType>(int_buf, p.merge_partial_packed_offset_end_offset);
    params.merge_partial_stride =
        offset_ptr<IdType>(int_buf, p.merge_partial_stride_offset);
    params.kv_indices =
        const_cast<IdType*>(reinterpret_cast<const IdType*>(kv_page_indices_d));
    params.q_len = offset_ptr<IdType>(int_buf, p.q_len_offset);
    params.kv_len = offset_ptr<IdType>(int_buf, p.kv_len_offset);
    params.q_start = offset_ptr<IdType>(int_buf, p.q_start_offset);
    params.kv_start = offset_ptr<IdType>(int_buf, p.kv_start_offset);
    params.kv_end = offset_ptr<IdType>(int_buf, p.kv_end_offset);
    params.work_indptr = offset_ptr<IdType>(int_buf, p.work_indptr_offset);

    params.block_size = ::flashinfer::uint_fastdiv(
        static_cast<std::uint32_t>(cache.page_size));
    params.num_heads = ::flashinfer::uint_fastdiv(
        static_cast<std::uint32_t>(cache.num_heads));

    params.q_nope_stride_n =
        static_cast<std::uint32_t>(cache.num_heads * cache.kv_lora_rank);
    params.q_nope_stride_h = static_cast<std::uint32_t>(cache.kv_lora_rank);
    params.q_pe_stride_n =
        static_cast<std::uint32_t>(cache.num_heads * cache.qk_rope_head_dim);
    params.q_pe_stride_h = static_cast<std::uint32_t>(cache.qk_rope_head_dim);
    params.ckv_stride_page =
        static_cast<std::uint32_t>(cache.page_size * cache.kv_lora_rank);
    params.ckv_stride_n = static_cast<std::uint32_t>(cache.kv_lora_rank);
    params.kpe_stride_page =
        static_cast<std::uint32_t>(cache.page_size * cache.qk_rope_head_dim);
    params.kpe_stride_n = static_cast<std::uint32_t>(cache.qk_rope_head_dim);
    params.o_stride_n =
        static_cast<std::uint32_t>(cache.num_heads * cache.kv_lora_rank);
    params.o_stride_h = static_cast<std::uint32_t>(cache.kv_lora_rank);
    params.sm_scale = cache.sm_scale;
    params.return_lse_base_on_e = true;

    CUDA_CHECK((::flashinfer::mla::BatchMLAPagedAttention<MASK, 512, 64>(
        params,
        static_cast<std::uint32_t>(p.num_blks_x),
        static_cast<std::uint32_t>(p.num_blks_y),
        stream)));
}

}  // namespace

// ── Naive paged MLA (Blackwell / sm100 fallback) ────────────────────────
// FlashInfer's FA2 BatchMLAPagedAttention (a cooperative kernel) produces
// zero output on sm_100; the ecosystem (sglang/vllm) routes Blackwell MLA to
// trtllm/cutlass/ragged kernels instead. This is a correctness-first,
// arch-agnostic latent-space MLA: one block per (token, head), flash-style
// online softmax over the paged ckv/kpe cache. Output is in the kv_lora
// latent space (same as the FA2 path), so the rest of the MLA forward
// (latent_to_v, o_proj) is unchanged.
namespace {

// The kernel + launcher live in a header so `driver/cuda/bench/mla_bench.cu`
// can compile the identical source standalone (seconds per iteration instead
// of a full engine rebuild).
using mla_naive::launch_mla_naive_paged_raw;

inline void launch_mla_naive_paged(
    const void* q_nope, const void* q_pe,
    const MlaCacheLayerView& layer, void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    int total_tokens, int num_requests, int num_heads,
    float sm_scale, bool causal, cudaStream_t stream,
    const std::uint8_t* index_mask, int index_mask_stride)
{
    launch_mla_naive_paged_raw(
        q_nope, q_pe, layer.ckv_pages, layer.kpe_pages,
        layer.kv_lora_rank, layer.qk_rope_head_dim, layer.page_size, o,
        qo_indptr_d, kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
        total_tokens, num_requests, num_heads, sm_scale, causal, stream,
        index_mask, index_mask_stride);
}

// Returns true if the naive MLA path should be used. Defaults to the device
// compute capability (Blackwell sm_100+ -> naive, since FlashInfer's FA2 MLA
// zero-outputs there); overridable via PIE_MLA_BACKEND=naive|fa2.
}  // namespace

void dispatch_attention_mla_bf16(
    const MlaPlanCache& cache,
    const void* q_nope,
    const void* q_pe,
    MlaCacheLayerView layer,
    void* o,
    const std::uint32_t* kv_page_indices_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    float* lse_out,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    const std::uint8_t* index_mask,
    int index_mask_stride)
{
    if (!cache.valid) {
        throw std::runtime_error(
            "dispatch_attention_mla_bf16: cache is empty; call plan first");
    }
    if (layer.kv_lora_rank != cache.kv_lora_rank ||
        layer.qk_rope_head_dim != cache.qk_rope_head_dim ||
        layer.page_size != cache.page_size) {
        throw std::runtime_error("flashinfer MLA: layer/cache shape mismatch");
    }
    if (mla_use_naive_backend()) {
        launch_mla_naive_paged(
            q_nope, q_pe, layer, o,
            qo_indptr_d, kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
            cache.total_tokens, cache.num_requests, cache.num_heads,
            cache.sm_scale, cache.causal, stream,
            index_mask, index_mask_stride);
        return;
    }
    if (cache.causal) {
        dispatch_mla_512_64<::flashinfer::MaskMode::kCausal>(
            cache, q_nope, q_pe, layer, o, kv_page_indices_d,
            workspace, stream, lse_out);
    } else {
        dispatch_mla_512_64<::flashinfer::MaskMode::kNone>(
            cache, q_nope, q_pe, layer, o, kv_page_indices_d,
            workspace, stream, lse_out);
    }
}

}  // namespace pie_cuda_driver::ops
