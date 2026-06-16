#include "ops/attention_mla.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include <cuda_bf16.h>
#include <math_constants.h>

#include <flashinfer/attention/mla.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/fastdiv.cuh>

#include "cuda_check.hpp"

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

constexpr int kMlaNaiveBlock = 128;

__global__ void mla_naive_paged_kernel(
    const __nv_bfloat16* __restrict__ q_nope,   // [N, H, CKV]
    const __nv_bfloat16* __restrict__ q_pe,     // [N, H, KPE]
    const __nv_bfloat16* __restrict__ ckv_pages,// [pages, page_size, CKV]
    const __nv_bfloat16* __restrict__ kpe_pages,// [pages, page_size, KPE]
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    __nv_bfloat16* __restrict__ o,              // [N, H, CKV]
    const std::uint8_t* __restrict__ index_mask, int index_mask_stride,
    int R, int H, int CKV, int KPE, int page_size, float sm_scale, bool causal)
{
    const int t = blockIdx.x;   // query token
    const int h = blockIdx.y;   // head
    const int tid = threadIdx.x;
    const int per = CKV / kMlaNaiveBlock;  // dims per thread in the latent acc
    // DSA mask row for this query (in-batch keys only).
    const std::uint8_t* mrow =
        (index_mask != nullptr) ? index_mask + static_cast<long long>(t) * index_mask_stride
                                : nullptr;

    // Resolve request, kv length, and this query's absolute position.
    int r = R - 1;
    for (int i = 0; i < R; ++i) {
        if (t < static_cast<int>(qo_indptr[i + 1])) { r = i; break; }
    }
    const int qo_lo = static_cast<int>(qo_indptr[r]);
    const int new_tokens = static_cast<int>(qo_indptr[r + 1]) - qo_lo;
    const int pages_first = static_cast<int>(kv_page_indptr[r]);
    const int num_pages = static_cast<int>(kv_page_indptr[r + 1]) - pages_first;
    const int kv_len =
        (num_pages - 1) * page_size + static_cast<int>(kv_last_page_lens[r]);
    const int pre_kv = kv_len - new_tokens;
    const int abs_q = pre_kv + (t - qo_lo);
    const int j_end = causal ? (abs_q + 1) : kv_len;

    extern __shared__ float smem[];
    float* qn_s = smem;            // CKV
    float* qp_s = smem + CKV;      // KPE
    float* red  = smem + CKV + KPE;// kMlaNaiveBlock

    const __nv_bfloat16* qn =
        q_nope + (static_cast<long long>(t) * H + h) * CKV;
    const __nv_bfloat16* qp =
        q_pe + (static_cast<long long>(t) * H + h) * KPE;
    for (int d = tid; d < CKV; d += kMlaNaiveBlock) qn_s[d] = __bfloat162float(qn[d]);
    for (int d = tid; d < KPE; d += kMlaNaiveBlock) qp_s[d] = __bfloat162float(qp[d]);
    __syncthreads();

    float acc[8];  // per <= 8 (CKV<=1024, BLOCK=128)
    for (int i = 0; i < per; ++i) acc[i] = 0.f;
    float m = -CUDART_INF_F, lsum = 0.f;

    for (int j = 0; j < j_end; ++j) {
        // DSA: skip keys not selected by the lightning indexer (in-batch only).
        if (mrow != nullptr && j < index_mask_stride && mrow[j] == 0) continue;
        const int page =
            static_cast<int>(kv_page_indices[pages_first + j / page_size]);
        const int off = j % page_size;
        const __nv_bfloat16* ckv_j =
            ckv_pages + (static_cast<long long>(page) * page_size + off) * CKV;
        const __nv_bfloat16* kpe_j =
            kpe_pages + (static_cast<long long>(page) * page_size + off) * KPE;
        float pd = 0.f;
        for (int i = 0; i < per; ++i) {
            const int d = tid + i * kMlaNaiveBlock;
            pd += qn_s[d] * __bfloat162float(ckv_j[d]);
        }
        if (tid < KPE) pd += qp_s[tid] * __bfloat162float(kpe_j[tid]);
        red[tid] = pd;
        __syncthreads();
        for (int s = kMlaNaiveBlock / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        const float score = red[0] * sm_scale;
        __syncthreads();
        const float m_new = fmaxf(m, score);
        const float corr = __expf(m - m_new);
        const float p = __expf(score - m_new);
        lsum = lsum * corr + p;
        for (int i = 0; i < per; ++i) {
            const int d = tid + i * kMlaNaiveBlock;
            acc[i] = acc[i] * corr + p * __bfloat162float(ckv_j[d]);
        }
        m = m_new;
    }

    __nv_bfloat16* out = o + (static_cast<long long>(t) * H + h) * CKV;
    const float inv = (lsum > 0.f) ? (1.f / lsum) : 0.f;
    for (int i = 0; i < per; ++i) {
        const int d = tid + i * kMlaNaiveBlock;
        out[d] = __float2bfloat16(acc[i] * inv);
    }
}

void launch_mla_naive_paged(
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
    if (total_tokens <= 0) return;
    if (qo_indptr_d == nullptr || kv_page_indptr_d == nullptr ||
        kv_last_page_lens_d == nullptr) {
        throw std::runtime_error(
            "naive MLA: missing device indptr/lens (qo/kv_page_indptr/"
            "kv_last_page_lens)");
    }
    const int CKV = layer.kv_lora_rank;
    const int KPE = layer.qk_rope_head_dim;
    if (CKV % kMlaNaiveBlock != 0 || CKV / kMlaNaiveBlock > 8) {
        throw std::runtime_error("naive MLA: unsupported kv_lora_rank");
    }
    const std::size_t smem =
        (static_cast<std::size_t>(CKV) + KPE + kMlaNaiveBlock) * sizeof(float);
    dim3 grid(total_tokens, num_heads);
    mla_naive_paged_kernel<<<grid, kMlaNaiveBlock, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(q_nope),
        static_cast<const __nv_bfloat16*>(q_pe),
        static_cast<const __nv_bfloat16*>(layer.ckv_pages),
        static_cast<const __nv_bfloat16*>(layer.kpe_pages),
        qo_indptr_d, kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
        static_cast<__nv_bfloat16*>(o),
        index_mask, index_mask_stride,
        num_requests, num_heads, CKV, KPE, layer.page_size, sm_scale, causal);
    CUDA_CHECK(cudaGetLastError());
}

// Returns true if the naive MLA path should be used. Defaults to the device
// compute capability (Blackwell sm_100+ -> naive, since FlashInfer's FA2 MLA
// zero-outputs there); overridable via PIE_MLA_BACKEND=naive|fa2.
bool mla_use_naive_backend() {
    static const int choice = [] {
        if (const char* e = std::getenv("PIE_MLA_BACKEND")) {
            if (std::strcmp(e, "naive") == 0) return 1;
            if (std::strcmp(e, "fa2") == 0) return 0;
        }
        int dev = 0;
        cudaGetDevice(&dev);
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        return major >= 10 ? 1 : 0;
    }();
    return choice == 1;
}

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
