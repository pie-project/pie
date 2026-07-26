#include "kernels/mla_paged.hpp"

#include <cuda_bf16.h>

#include "cuda_check.hpp"
#include "kernels/rope_device.cuh"

namespace pie_cuda_driver::kernels {

// 32 KB of shared would cap this far higher; every MLA model here uses 64.
constexpr int kMaxRopePairs = 128;

namespace {

__device__ __forceinline__ int find_request(const std::uint32_t* qo_indptr,
                                            int R,
                                            int token_idx) {
    for (int r = 0; r < R; ++r) {
        if (token_idx < static_cast<int>(qo_indptr[r + 1])) return r;
    }
    return R - 1;
}

__device__ __forceinline__ void resolve_dst(
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    int R,
    int page_size,
    int token_idx,
    int& actual_page,
    int& offset_in_page)
{
    const int r = find_request(qo_indptr, R, token_idx);
    const int qo_lo = qo_indptr[r];
    const int qo_hi = qo_indptr[r + 1];
    const int new_tokens_r = qo_hi - qo_lo;
    const int offset_in_new = token_idx - qo_lo;
    const int pages_first = kv_page_indptr[r];
    const int pages_last = kv_page_indptr[r + 1];
    const int num_pages_r = pages_last - pages_first;
    const int total_kv_after =
        (num_pages_r - 1) * page_size + kv_last_page_lens[r];
    const int pre_kv_len = total_kv_after - new_tokens_r;
    const int abs_kv_pos = pre_kv_len + offset_in_new;
    const int page_in_req = abs_kv_pos / page_size;
    offset_in_page = abs_kv_pos % page_size;
    actual_page = static_cast<int>(kv_page_indices[pages_first + page_in_req]);
}

__global__ void write_mla_kernel(
    const __nv_bfloat16* __restrict__ ckv_curr,
    const __nv_bfloat16* __restrict__ kpe_curr,
    __nv_bfloat16* __restrict__ ckv_pages,
    __nv_bfloat16* __restrict__ kpe_pages,
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    const std::uint8_t* __restrict__ row_valid,
    int R,
    int page_size,
    int kv_lora_rank,
    int qk_rope_head_dim)
{
    const int t = blockIdx.x;
    if (row_valid != nullptr && row_valid[t] == 0) return;
    int actual_page = 0;
    int offset_in_page = 0;
    resolve_dst(qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                R, page_size, t, actual_page, offset_in_page);

    const long long ckv_src = static_cast<long long>(t) * kv_lora_rank;
    const long long ckv_dst =
        (static_cast<long long>(actual_page) * page_size + offset_in_page) *
        kv_lora_rank;
    for (int i = threadIdx.x; i < kv_lora_rank; i += blockDim.x) {
        ckv_pages[ckv_dst + i] = ckv_curr[ckv_src + i];
    }

    const long long kpe_src = static_cast<long long>(t) * qk_rope_head_dim;
    const long long kpe_dst =
        (static_cast<long long>(actual_page) * page_size + offset_in_page) *
        qk_rope_head_dim;
    for (int i = threadIdx.x; i < qk_rope_head_dim; i += blockDim.x) {
        kpe_pages[kpe_dst + i] = kpe_curr[kpe_src + i];
    }
}

// Fused MLA prologue: split_kv_a+rmsnorm, split_q_b, RoPE and the paged-cache
// write in one launch.
//
// The four kernels this replaces are all per-token with no cross-token
// dependency, and at decode shapes each one costs about what an empty kernel
// costs -- 70 us/step of which roughly 62 us was launch latency and inter-kernel
// gap, against ~8 us of arithmetic. Nothing here is faster than it was; there
// is simply one launch instead of four.
//
// Layout: blockIdx.x is the token. blockIdx.y == 0 handles the KV lane (norm,
// k_pe rotation, page write); blockIdx.y >= 1 splits the query heads. The KV
// lane must own both the k_pe rotation and the page write, because the write
// consumes the rotated value and a cross-block dependency inside one kernel
// would need a grid sync.
template <int BLOCK_DIM>
__global__ void mla_prepare_kernel(
    const __nv_bfloat16* __restrict__ kv_a,
    const __nv_bfloat16* __restrict__ kv_a_norm_w,
    const __nv_bfloat16* __restrict__ q_b,
    __nv_bfloat16* __restrict__ kv_c,
    __nv_bfloat16* __restrict__ k_pe,
    __nv_bfloat16* __restrict__ q_nope,
    __nv_bfloat16* __restrict__ q_pe,
    __nv_bfloat16* __restrict__ ckv_pages,
    __nv_bfloat16* __restrict__ kpe_pages,
    const std::int32_t* __restrict__ positions,
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    const std::uint8_t* __restrict__ row_valid,
    int R,
    int page_size,
    int heads,
    int kv_lora,
    int nope,
    int rope,
    int src_row_stride,
    float eps,
    float theta,
    bool interleaved,
    int heads_per_block,
    // Original-YaRN scaling (Kimi/DeepSeek). factor <= 0 selects plain RoPE.
    float yarn_factor,
    float yarn_low_dim,
    float yarn_high_dim,
    float yarn_mscale)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const int half = rope / 2;
    const int pos = positions[n];

    // The angle depends only on (pos, dim_pair), so every head of this token
    // shares it. rope is 64 for every MLA model here, so 32 pairs; a static
    // array avoids the "two extern __shared__ of different types in one TU"
    // problem and costs 512 B.
    __shared__ float cs[2 * kMaxRopePairs];
    const int cached = half <= kMaxRopePairs ? half : 0;
    const bool yarn = yarn_factor > 0.f;
    auto angle = [&](int dp, float& c, float& s_) {
        if (yarn) {
            rope_cos_sin_yarn_original(theta, dp, rope, pos, yarn_factor,
                                       yarn_low_dim, yarn_high_dim,
                                       yarn_mscale, c, s_);
        } else {
            rope_cos_sin(theta, dp, rope, pos, c, s_);
        }
    };
    for (int dp = tid; dp < cached; dp += BLOCK_DIM) {
        angle(dp, cs[dp], cs[cached + dp]);
    }
    if (cached > 0) __syncthreads();

    if (blockIdx.y == 0) {
        const __nv_bfloat16* row =
            kv_a + static_cast<long long>(n) * src_row_stride;
        __nv_bfloat16* kpe_out = k_pe + static_cast<long long>(n) * rope;

        // k_pe: copy out of the projection and rotate in one pass. The
        // standalone pair wrote the unrotated value to memory and read it back.
        for (int dp = tid; dp < half; dp += BLOCK_DIM) {
            float cos_v, sin_v;
            if (dp < cached) { cos_v = cs[dp]; sin_v = cs[cached + dp]; }
            else angle(dp, cos_v, sin_v);
            if (interleaved) {
                rotate_pair_interleaved_to(row + kv_lora, kpe_out, dp, cos_v, sin_v);
            } else {
                rotate_pair_to(row + kv_lora, kpe_out, half, dp, cos_v, sin_v);
            }
        }

        // RMSNorm over the latent half. Same reduction tree and same block
        // width as `split_kv_a_norm_kernel`, so the sum is bit-identical.
        float local = 0.f;
        for (int d = tid; d < kv_lora; d += BLOCK_DIM) {
            const float v = __bfloat162float(row[d]);
            local += v * v;
        }
        __shared__ float buf[BLOCK_DIM];
        buf[tid] = local;
        __syncthreads();
        for (int off = BLOCK_DIM / 2; off > 0; off >>= 1) {
            if (tid < off) buf[tid] += buf[tid + off];
            __syncthreads();
        }
        const float inv_rms =
            rsqrtf(buf[0] / static_cast<float>(kv_lora) + eps);

        __nv_bfloat16* kvc_out = kv_c + static_cast<long long>(n) * kv_lora;
        for (int d = tid; d < kv_lora; d += BLOCK_DIM) {
            const float v = __bfloat162float(row[d]);
            const float w = __bfloat162float(kv_a_norm_w[d]);
            kvc_out[d] = __float2bfloat16(v * inv_rms * w);
        }

        if (row_valid != nullptr && row_valid[n] == 0) return;
        __syncthreads();

        int actual_page = 0;
        int offset_in_page = 0;
        resolve_dst(qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, R, page_size, n,
                    actual_page, offset_in_page);
        const long long slot =
            static_cast<long long>(actual_page) * page_size + offset_in_page;
        __nv_bfloat16* ckv_dst = ckv_pages + slot * kv_lora;
        for (int d = tid; d < kv_lora; d += BLOCK_DIM) ckv_dst[d] = kvc_out[d];
        __nv_bfloat16* kpe_dst = kpe_pages + slot * rope;
        for (int d = tid; d < rope; d += BLOCK_DIM) kpe_dst[d] = kpe_out[d];
        return;
    }

    // Query lane: split q_b into (q_nope, q_pe) and rotate q_pe.
    const int per = nope + rope;
    const int head_base = (blockIdx.y - 1) * heads_per_block;
    const int heads_here = min(heads_per_block, heads - head_base);
    if (heads_here <= 0) return;

    const __nv_bfloat16* qb_row =
        q_b + (static_cast<long long>(n) * heads + head_base) * per;
    __nv_bfloat16* qn_row =
        q_nope + (static_cast<long long>(n) * heads + head_base) * nope;
    for (int i = tid; i < heads_here * nope; i += BLOCK_DIM) {
        const int h = i / nope;
        qn_row[i] = qb_row[static_cast<long long>(h) * per + (i - h * nope)];
    }

    __nv_bfloat16* qp_row =
        q_pe + (static_cast<long long>(n) * heads + head_base) * rope;
    for (int i = tid; i < heads_here * half; i += BLOCK_DIM) {
        const int h = i / half;
        const int dp = i - h * half;
        float cos_v, sin_v;
        if (dp < cached) { cos_v = cs[dp]; sin_v = cs[cached + dp]; }
        else angle(dp, cos_v, sin_v);
        const __nv_bfloat16* src =
            qb_row + static_cast<long long>(h) * per + nope;
        __nv_bfloat16* dst = qp_row + static_cast<long long>(h) * rope;
        if (interleaved) rotate_pair_interleaved_to(src, dst, dp, cos_v, sin_v);
        else rotate_pair_to(src, dst, half, dp, cos_v, sin_v);
    }
}

}  // namespace

bool mla_prepare_supported(int qk_rope_head_dim) {
    return qk_rope_head_dim > 0 && qk_rope_head_dim % 2 == 0 &&
           qk_rope_head_dim / 2 <= kMaxRopePairs;
}

void launch_mla_prepare_bf16(
    MlaCacheLayerView layer,
    const void* kv_a,
    const void* kv_a_norm_weight,
    const void* q_b,
    void* kv_c,
    void* k_pe,
    void* q_nope,
    void* q_pe,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    int total_tokens,
    int num_requests,
    int heads,
    int qk_nope_head_dim,
    float eps,
    float theta,
    bool interleaved,
    int kv_a_row_stride,
    const YarnOriginalParams* yarn,
    cudaStream_t stream,
    const std::uint8_t* row_valid)
{
    if (total_tokens <= 0) return;
    constexpr int BS = 256;
    const int kv_lora = layer.kv_lora_rank;
    const int rope = layer.qk_rope_head_dim;
    const int half = rope / 2;
    const int stride =
        kv_a_row_stride > 0 ? kv_a_row_stride : kv_lora + rope;
    // Match `launch_rope_bf16`'s head packing so the query lane has the same
    // shape of work per block that the standalone kernel had.
    const int heads_per_block = half >= BS ? 1 : (BS / half);
    const int q_blocks = (heads + heads_per_block - 1) / heads_per_block;
    float low_dim = 0.f, high_dim = 0.f;
    if (yarn != nullptr) {
        yarn_original_ramp_bounds(rope, theta, yarn->beta_fast,
                                  yarn->beta_slow,
                                  yarn->original_max_position,
                                  low_dim, high_dim);
    }
    dim3 grid(total_tokens, 1 + q_blocks);
    mla_prepare_kernel<BS><<<grid, BS, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(kv_a),
        static_cast<const __nv_bfloat16*>(kv_a_norm_weight),
        static_cast<const __nv_bfloat16*>(q_b),
        static_cast<__nv_bfloat16*>(kv_c),
        static_cast<__nv_bfloat16*>(k_pe),
        static_cast<__nv_bfloat16*>(q_nope),
        static_cast<__nv_bfloat16*>(q_pe),
        static_cast<__nv_bfloat16*>(layer.ckv_pages),
        static_cast<__nv_bfloat16*>(layer.kpe_pages),
        positions, qo_indptr, kv_page_indices, kv_page_indptr,
        kv_last_page_lens, row_valid,
        num_requests, layer.page_size, heads, kv_lora, qk_nope_head_dim, rope,
        stride, eps, theta, interleaved, heads_per_block,
        yarn != nullptr ? yarn->factor : -1.f, low_dim, high_dim,
        yarn != nullptr ? yarn->attention_factor : 1.f);
    CUDA_CHECK(cudaGetLastError());
}

void launch_write_mla_to_pages_bf16(
    void* ckv_pages,
    void* kpe_pages,
    const void* ckv_curr,
    const void* kpe_curr,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    int total_tokens,
    int num_requests,
    int page_size,
    int kv_lora_rank,
    int qk_rope_head_dim,
    cudaStream_t stream,
    const std::uint8_t* row_valid)
{
    if (total_tokens <= 0) return;
    write_mla_kernel<<<total_tokens, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(ckv_curr),
        static_cast<const __nv_bfloat16*>(kpe_curr),
        static_cast<__nv_bfloat16*>(ckv_pages),
        static_cast<__nv_bfloat16*>(kpe_pages),
        qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
        row_valid,
        num_requests, page_size, kv_lora_rank, qk_rope_head_dim);
    CUDA_CHECK(cudaGetLastError());
}

void launch_write_mla_to_pages(
    MlaCacheLayerView layer,
    const void* ckv_curr,
    const void* kpe_curr,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    int total_tokens,
    int num_requests,
    cudaStream_t stream,
    const std::uint8_t* row_valid)
{
    launch_write_mla_to_pages_bf16(
        layer.ckv_pages, layer.kpe_pages, ckv_curr, kpe_curr,
        qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
        total_tokens, num_requests, layer.page_size, layer.kv_lora_rank,
        layer.qk_rope_head_dim, stream, row_valid);
}

}  // namespace pie_cuda_driver::kernels
