#include "mla_paged.cuh"

#include <cstdint>
#include <stdexcept>

#include <cuda_bf16.h>
#include <math_constants.h>

// Lifted VERBATIM from driver/cuda/src/ops/attention_mla.cu — the raw-pointer
// bf16 naive MLA path only (`mla_naive_paged_kernel` + `launch_mla_naive_paged`,
// renamed to `mla_naive_paged` to match the device-library convention). The
// changes are:
//   - namespace `pie_cuda_driver::ops` -> `pie_cuda_device::ops`;
//   - the `launch_` prefix dropped from the entry point;
//   - the `MlaCacheLayerView layer` argument replaced by raw device pointers
//     (`ckv_pages`, `kpe_pages`) plus the scalar dims it unpacked
//     (`kv_lora_rank`, `qk_rope_head_dim`, `page_size`), mirroring how
//     attention_naive_paged.{cuh,cu} replaced KvCacheLayerView with raw
//     pointers — so this TU never includes mla_cache.hpp / tensor.hpp;
//   - the post-launch `CUDA_CHECK(cudaGetLastError())` dropped (the ABI layer
//     centralizes that, mirroring kernels/rmsnorm.cu and
//     ops/attention_naive_paged.cu), so this TU never includes cuda_check.hpp.
// The flashinfer FA2 path (plan / dispatch / MlaPlanCache / backend selector)
// is not lifted, so this TU never includes any flashinfer header.
//
// The kernel body itself — the NoPE+RoPE score split, the online softmax over
// the compressed-latent ckv/kpe pages, and the latent-space output — is
// unchanged.

namespace pie_cuda_device::ops {
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

}  // namespace

void mla_naive_paged(
    const void* q_nope,
    const void* q_pe,
    const void* ckv_pages,
    const void* kpe_pages,
    void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    int total_tokens,
    int num_requests,
    int num_heads,
    int kv_lora_rank,
    int qk_rope_head_dim,
    int page_size,
    float sm_scale,
    bool causal,
    cudaStream_t stream,
    const std::uint8_t* index_mask,
    int index_mask_stride)
{
    if (total_tokens <= 0) return;
    if (qo_indptr_d == nullptr || kv_page_indptr_d == nullptr ||
        kv_last_page_lens_d == nullptr) {
        throw std::runtime_error(
            "naive MLA: missing device indptr/lens (qo/kv_page_indptr/"
            "kv_last_page_lens)");
    }
    const int CKV = kv_lora_rank;
    const int KPE = qk_rope_head_dim;
    if (CKV % kMlaNaiveBlock != 0 || CKV / kMlaNaiveBlock > 8) {
        throw std::runtime_error("naive MLA: unsupported kv_lora_rank");
    }
    const std::size_t smem =
        (static_cast<std::size_t>(CKV) + KPE + kMlaNaiveBlock) * sizeof(float);
    dim3 grid(total_tokens, num_heads);
    mla_naive_paged_kernel<<<grid, kMlaNaiveBlock, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(q_nope),
        static_cast<const __nv_bfloat16*>(q_pe),
        static_cast<const __nv_bfloat16*>(ckv_pages),
        static_cast<const __nv_bfloat16*>(kpe_pages),
        qo_indptr_d, kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
        static_cast<__nv_bfloat16*>(o),
        index_mask, index_mask_stride,
        num_requests, num_heads, CKV, KPE, page_size, sm_scale, causal);
}

}  // namespace pie_cuda_device::ops
