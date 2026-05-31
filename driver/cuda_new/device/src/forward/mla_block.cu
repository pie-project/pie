#include "mla_block.cuh"

#include "../kernels/mla_write.cuh"
#include "../kernels/residual_add.cuh"
#include "../kernels/rmsnorm.cuh"
#include "../kernels/rope_partial.cuh"
#include "../ops/gemm.cuh"
#include "../ops/mla_paged.cuh"

#include <cuda_bf16.h>

namespace pie_cuda_device::forward {

namespace {

// q_proj rows are [T, nh, qk_nope+qk_rope] (head-major within a token). Split
// into a packed NoPE buffer [T, nh, qk_nope] and a packed RoPE buffer
// [T, nh, qk_rope]. One block per (token, head); threads cover the head dim.
__global__ void split_q_kernel(
    const __nv_bfloat16* __restrict__ q,   // [T, nh*(nope+rope)]
    __nv_bfloat16* __restrict__ q_nope,    // [T, nh, nope]
    __nv_bfloat16* __restrict__ q_pe,      // [T, nh, rope]
    int nh, int nope, int rope) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int per_head = nope + rope;
    const __nv_bfloat16* src =
        q + (static_cast<long long>(t) * nh + h) * per_head;
    __nv_bfloat16* dn = q_nope + (static_cast<long long>(t) * nh + h) * nope;
    __nv_bfloat16* dp = q_pe + (static_cast<long long>(t) * nh + h) * rope;
    for (int i = threadIdx.x; i < nope; i += blockDim.x) dn[i] = src[i];
    for (int i = threadIdx.x; i < rope; i += blockDim.x) dp[i] = src[nope + i];
}

// kv_a rows are [T, kv_lora_rank + qk_rope]. Split into compressed-kv latent
// [T, kv_lora_rank] and the shared RoPE key [T, qk_rope]. One block per token.
__global__ void split_kv_kernel(
    const __nv_bfloat16* __restrict__ kv_a,  // [T, ckv+rope]
    __nv_bfloat16* __restrict__ ckv_out,     // [T, ckv]
    __nv_bfloat16* __restrict__ k_pe,        // [T, rope]
    int ckv, int rope) {
    const int t = blockIdx.x;
    const int per = ckv + rope;
    const __nv_bfloat16* src = kv_a + static_cast<long long>(t) * per;
    __nv_bfloat16* dc = ckv_out + static_cast<long long>(t) * ckv;
    __nv_bfloat16* dp = k_pe + static_cast<long long>(t) * rope;
    for (int i = threadIdx.x; i < ckv; i += blockDim.x) dc[i] = src[i];
    for (int i = threadIdx.x; i < rope; i += blockDim.x) dp[i] = src[ckv + i];
}

// Gather one head's contiguous [T, D] slice out of a packed [T, nh, D] tensor.
__global__ void gather_head_kernel(
    const __nv_bfloat16* __restrict__ src,  // [T, nh, D]
    __nv_bfloat16* __restrict__ dst,        // [T, D]
    int nh, int head, int D) {
    const int t = blockIdx.x;
    const __nv_bfloat16* s = src + (static_cast<long long>(t) * nh + head) * D;
    __nv_bfloat16* d = dst + static_cast<long long>(t) * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) d[i] = s[i];
}

// Scatter a contiguous [T, D] slice into head `head` of a packed [T, nh, D].
__global__ void scatter_head_kernel(
    const __nv_bfloat16* __restrict__ src,  // [T, D]
    __nv_bfloat16* __restrict__ dst,        // [T, nh, D]
    int nh, int head, int D) {
    const int t = blockIdx.x;
    const __nv_bfloat16* s = src + static_cast<long long>(t) * D;
    __nv_bfloat16* d = dst + (static_cast<long long>(t) * nh + head) * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) d[i] = s[i];
}

}  // namespace

cudaError_t mla_block_bf16(
    cublasHandle_t cublas, cudaStream_t stream,
    void* hidden, const MlaLayerWeights& w, const std::int32_t* positions,
    void* ckv_pages, void* kpe_pages,
    const std::uint32_t* qo_indptr, const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr, const std::uint32_t* kv_last_page_lens,
    int num_tokens, int num_requests, int H, int num_heads,
    int q_lora_rank, int kv_lora_rank,
    int qk_nope_head_dim, int qk_rope_head_dim, int v_head_dim,
    int page_size, float rms_eps, float sm_scale, float rope_theta) {
    const int T = num_tokens;
    const int nh = num_heads;
    const int nope = qk_nope_head_dim;
    const int rope = qk_rope_head_dim;
    const int ckv = kv_lora_rank;
    const int qk = nope + rope;             // per-head q width
    const int q_b_out = nh * qk;            // W_q_b output width
    const int kv_a_out = ckv + rope;        // W_kv_a output width
    const int ov_width = nh * v_head_dim;   // o_proj input width
    constexpr size_t es = sizeof(std::uint16_t);  // bf16

    // Scratch buffers (device bf16). Sized for the token capacity, freed at end.
    enum {
        HN,         // [T, H]              rmsnorm(hidden)
        Q_A,        // [T, q_lora_rank]    q down-proj (also reused post-norm)
        Q_B,        // [T, nh*qk]          q up-proj (head-major nope|rope)
        Q_NOPE,     // [T, nh, nope]       packed NoPE query
        Q_PE,       // [T, nh, rope]       packed RoPE query
        Q_LAT,      // [T, nh, ckv]        absorbed NoPE query (latent)
        KV_A,       // [T, ckv+rope]       kv down-proj
        CKV,        // [T, ckv]            compressed-kv latent (post kv-norm)
        K_PE,       // [T, rope]           shared RoPE key (one head)
        O_LAT,      // [T, nh, ckv]        latent attention output
        O_V,        // [T, nh, v_head_dim] absorbed value output (packed)
        O_PROJ,     // [T, H]              o_proj output
        HEAD_IN,    // [T, max(nope,ckv)]  per-head gemm input (gather target)
        HEAD_OUT,   // [T, max(ckv,vhd)]   per-head gemm output (scatter source)
        NBUF
    };
    const int head_in_w = nope > ckv ? nope : ckv;
    const int head_out_w = ckv > v_head_dim ? ckv : v_head_dim;
    const size_t sizes[NBUF] = {
        (size_t)T * H * es,
        (size_t)T * q_lora_rank * es,
        (size_t)T * q_b_out * es,
        (size_t)T * nh * nope * es,
        (size_t)T * nh * rope * es,
        (size_t)T * nh * ckv * es,
        (size_t)T * kv_a_out * es,
        (size_t)T * ckv * es,
        (size_t)T * rope * es,
        (size_t)T * nh * ckv * es,
        (size_t)T * ov_width * es,
        (size_t)T * H * es,
        (size_t)T * head_in_w * es,
        (size_t)T * head_out_w * es,
    };
    void* b[NBUF] = {};
    for (int i = 0; i < NBUF; ++i) {
        if (cudaError_t e = cudaMalloc(&b[i], sizes[i]); e != cudaSuccess) {
            for (int j = 0; j < i; ++j) cudaFree(b[j]);
            return e;
        }
    }

    auto bf = [](void* p) { return static_cast<__nv_bfloat16*>(p); };
    auto cbf = [](const void* p) { return static_cast<const __nv_bfloat16*>(p); };
    constexpr int BLK = 128;

    // 1. RMSNorm(hidden) -> hn
    kernels::rmsnorm_bf16(hidden, w.attn_norm, b[HN], T, H, rms_eps, stream);

    // 2. Q path (lora): q_a = hn @ W_q_a^T ; rmsnorm(q_a) ; q = q_a @ W_q_b^T
    ops::gemm_act_x_wt_bf16(cublas, b[HN], w.W_q_a, b[Q_A], T, q_lora_rank, H, 0.f);
    kernels::rmsnorm_bf16(b[Q_A], w.q_a_ln, b[Q_A], T, q_lora_rank, rms_eps, stream);
    ops::gemm_act_x_wt_bf16(cublas, b[Q_A], w.W_q_b, b[Q_B], T, q_b_out, q_lora_rank, 0.f);
    // split q -> q_nope [T,nh,nope], q_pe [T,nh,rope]
    split_q_kernel<<<dim3(T, nh), BLK, 0, stream>>>(
        cbf(b[Q_B]), bf(b[Q_NOPE]), bf(b[Q_PE]), nh, nope, rope);

    // 3. KV path: kv_a = hn @ W_kv_a^T ; split ; rmsnorm(compressed_kv)
    ops::gemm_act_x_wt_bf16(cublas, b[HN], w.W_kv_a, b[KV_A], T, kv_a_out, H, 0.f);
    split_kv_kernel<<<dim3(T), BLK, 0, stream>>>(
        cbf(b[KV_A]), bf(b[CKV]), bf(b[K_PE]), ckv, rope);
    kernels::rmsnorm_bf16(b[CKV], w.kv_a_ln, b[CKV], T, ckv, rms_eps, stream);

    // 4. RoPE on q_pe (nh heads) and k_pe (1 shared head), full rope slice.
    kernels::rope_partial_bf16(b[Q_PE], b[K_PE], positions, T, nh, /*kv heads*/ 1,
                               rope, /*rotary_dim=*/rope, rope_theta, stream);

    // 5. Absorb: q_nope_latent[:,h,:] = q_nope[:,h,:] @ W_uk[h]   per head.
    //    W_uk[h] is [ckv, nope] (transposed), used as gemm `w` (act@w^T):
    //    out[t,l] = sum_d q_nope[t,d] * W_uk[h][l,d] = q_nope[t,:] @ W_uk_orig[h].
    for (int h = 0; h < nh; ++h) {
        gather_head_kernel<<<T, BLK, 0, stream>>>(cbf(b[Q_NOPE]), bf(b[HEAD_IN]),
                                                  nh, h, nope);
        const void* W_uk_h = cbf(w.W_uk) + (size_t)h * ckv * nope;
        ops::gemm_act_x_wt_bf16(cublas, b[HEAD_IN], W_uk_h, b[HEAD_OUT],
                                T, ckv, nope, 0.f);
        scatter_head_kernel<<<T, BLK, 0, stream>>>(cbf(b[HEAD_OUT]), bf(b[Q_LAT]),
                                                   nh, h, ckv);
    }

    // 6. Write compressed_kv -> ckv_pages, k_pe -> kpe_pages.
    kernels::write_mla_to_pages_bf16(ckv_pages, kpe_pages, b[CKV], b[K_PE],
                                     qo_indptr, kv_page_indices, kv_page_indptr,
                                     kv_last_page_lens, T, num_requests, page_size,
                                     ckv, rope, stream);

    // 7. Latent paged MLA attention -> o_latent [T,nh,ckv].
    ops::mla_naive_paged(b[Q_LAT], b[Q_PE], ckv_pages, kpe_pages, b[O_LAT],
                         qo_indptr, kv_page_indices, kv_page_indptr,
                         kv_last_page_lens, T, num_requests, nh, ckv, rope,
                         page_size, sm_scale, /*causal=*/true, stream);

    // 8a. Absorb V up-proj: o_v[:,h,:] = o_latent[:,h,:] @ W_uv[h]  per head.
    //     W_uv[h] is [v_head_dim, ckv] (transposed), used as gemm `w`:
    //     out[t,vd] = sum_l o_latent[t,l] * W_uv[h][vd,l].
    for (int h = 0; h < nh; ++h) {
        gather_head_kernel<<<T, BLK, 0, stream>>>(cbf(b[O_LAT]), bf(b[HEAD_IN]),
                                                  nh, h, ckv);
        const void* W_uv_h = cbf(w.W_uv) + (size_t)h * v_head_dim * ckv;
        ops::gemm_act_x_wt_bf16(cublas, b[HEAD_IN], W_uv_h, b[HEAD_OUT],
                                T, v_head_dim, ckv, 0.f);
        scatter_head_kernel<<<T, BLK, 0, stream>>>(cbf(b[HEAD_OUT]), bf(b[O_V]),
                                                   nh, h, v_head_dim);
    }

    // 8b. out = reshape(o_v)[T, nh*v_head_dim] @ W_o^T -> [T, H]
    ops::gemm_act_x_wt_bf16(cublas, b[O_V], w.W_o, b[O_PROJ], T, H, ov_width, 0.f);

    // 9. residual add to the input hidden.
    kernels::residual_add_bf16(hidden, b[O_PROJ], (size_t)T * H, stream);

    cudaError_t e = cudaStreamSynchronize(stream);
    for (int i = 0; i < NBUF; ++i) cudaFree(b[i]);
    return e;
}

}  // namespace pie_cuda_device::forward
