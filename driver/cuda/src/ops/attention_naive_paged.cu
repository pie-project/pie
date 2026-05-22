#include "ops/attention_naive_paged.hpp"

#include <cmath>

#include <cuda_bf16.h>

#include "kernels/kv_paged.hpp"

namespace pie_cuda_driver::ops {

namespace {

constexpr int BLOCK = 128;

// One block per (request_idx, qo_offset, q_head). Threads cover the
// `head_dim` axis (`head_dim ≤ 1024` keeps us under 1 thread per dim).
//
// The kernel loops over the request's pages, two-pass softmax: the
// first pass computes `row_max = max_{kv} q·k_kv * sm_scale`, the
// second accumulates `Σ exp(q·k - row_max) * v` and the matching
// denominator. We use fp32 throughout for the partial sums because
// the per-pass passes touch O(head_dim · kv_len) bf16 reads — keeping
// the reduction in fp32 avoids the catastrophic cancellation you'd
// see in bf16 for long contexts.
__global__ void naive_paged_attn_bf16_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_pages,
    const __nv_bfloat16* __restrict__ v_pages,
    __nv_bfloat16*       __restrict__ o,
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    int num_q_heads, int num_kv_heads,
    int head_dim, int page_size,
    int window_left, float sm_scale)
{
    const int r        = blockIdx.x;          // request idx
    const int qo_off   = blockIdx.y;          // offset within this request
    const int q_head   = blockIdx.z;          // q head idx
    const int tid      = threadIdx.x;
    const int kv_head  = q_head / (num_q_heads / num_kv_heads);

    const std::uint32_t qo_lo  = qo_indptr[r];
    const std::uint32_t qo_hi  = qo_indptr[r + 1];
    if (qo_off >= int(qo_hi - qo_lo)) return;
    const int qo_global = static_cast<int>(qo_lo) + qo_off;

    const std::uint32_t pg_lo = kv_page_indptr[r];
    const std::uint32_t pg_hi = kv_page_indptr[r + 1];
    const int last_page_len   = static_cast<int>(kv_last_page_lens[r]);
    const int num_full_pages  = static_cast<int>(pg_hi - pg_lo) - 1;
    const int kv_total        = (num_full_pages > 0)
                                    ? num_full_pages * page_size + last_page_len
                                    : last_page_len;

    // Causal: a query at offset `qo_off` from this request's start
    // sees the first `kv_total - (qo_hi - qo_lo) + qo_off + 1` KV
    // rows. The trailing `(qo_hi - qo_lo)` KV rows correspond to this
    // fire's own queries.
    const int qo_len  = static_cast<int>(qo_hi - qo_lo);
    const int kv_lim  = kv_total - qo_len + qo_off + 1;

    // Q row pointer.
    const __nv_bfloat16* q_row =
        q + (static_cast<long long>(qo_global) * num_q_heads + q_head) * head_dim;

    extern __shared__ float smem[];
    float* q_smem = smem;                     // [head_dim] q values, fp32
    float* reduce = smem + head_dim;          // [BLOCK] reduction scratch

    // Stage Q into shared memory in fp32.
    for (int d = tid; d < head_dim; d += BLOCK) {
        q_smem[d] = __bfloat162float(q_row[d]);
    }
    __syncthreads();

    const float scale = (sm_scale > 0.f) ? sm_scale
                        : (1.0f / sqrtf(static_cast<float>(head_dim)));

    // ── Pass 1: row_max ──
    float local_max = -INFINITY;
    for (int kv = tid; kv < kv_lim; kv += BLOCK) {
        // Sliding window check.
        if (window_left >= 0 && kv < kv_lim - 1 - window_left) continue;
        const int page_idx = kv / page_size;
        const int slot     = kv % page_size;
        const std::uint32_t pg_id = kv_page_indices[pg_lo + page_idx];
        const __nv_bfloat16* k_row =
            k_pages +
            ((static_cast<long long>(pg_id) * page_size + slot) * num_kv_heads + kv_head) *
                head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_smem[d] * __bfloat162float(k_row[d]);
        }
        dot *= scale;
        if (dot > local_max) local_max = dot;
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) reduce[tid] = fmaxf(reduce[tid], reduce[tid + off]);
        __syncthreads();
    }
    const float row_max = reduce[0];
    __syncthreads();

    // ── Pass 2: exp + V weighted sum ──
    // Each thread owns a slice of the head_dim output dims; we
    // accumulate across kv rows in registers.
    const int dims_per_thread = (head_dim + BLOCK - 1) / BLOCK;
    float acc[8];                             // upper bound: 1024/128
    for (int i = 0; i < dims_per_thread; ++i) acc[i] = 0.f;

    float local_z = 0.f;
    for (int kv = 0; kv < kv_lim; ++kv) {
        if (window_left >= 0 && kv < kv_lim - 1 - window_left) continue;
        const int page_idx = kv / page_size;
        const int slot     = kv % page_size;
        const std::uint32_t pg_id = kv_page_indices[pg_lo + page_idx];
        const __nv_bfloat16* k_row =
            k_pages +
            ((static_cast<long long>(pg_id) * page_size + slot) * num_kv_heads + kv_head) *
                head_dim;
        const __nv_bfloat16* v_row =
            v_pages +
            ((static_cast<long long>(pg_id) * page_size + slot) * num_kv_heads + kv_head) *
                head_dim;

        // Cooperatively compute q·k across threads, reducing through
        // shared memory. This pass dominates runtime — the reduction
        // is BLOCK-wide for every KV row.
        float partial = 0.f;
        for (int d = tid; d < head_dim; d += BLOCK) {
            partial += q_smem[d] * __bfloat162float(k_row[d]);
        }
        reduce[tid] = partial;
        __syncthreads();
        for (int off = BLOCK / 2; off > 0; off >>= 1) {
            if (tid < off) reduce[tid] += reduce[tid + off];
            __syncthreads();
        }
        const float dot   = reduce[0] * scale;
        const float w     = expf(dot - row_max);
        if (tid == 0) local_z += w;
        __syncthreads();
        // Accumulate V*w into per-thread `acc[i]` for this thread's
        // slice of head_dim.
        for (int i = 0; i < dims_per_thread; ++i) {
            const int d = tid + i * BLOCK;
            if (d < head_dim) {
                acc[i] = fmaf(__bfloat162float(v_row[d]), w, acc[i]);
            }
        }
    }

    // Broadcast `local_z` (only thread 0 has it) to every thread.
    __shared__ float z_shared;
    if (tid == 0) z_shared = local_z;
    __syncthreads();
    const float inv_z = 1.0f / z_shared;

    __nv_bfloat16* o_row =
        o + (static_cast<long long>(qo_global) * num_q_heads + q_head) * head_dim;
    for (int i = 0; i < dims_per_thread; ++i) {
        const int d = tid + i * BLOCK;
        if (d < head_dim) {
            o_row[d] = __float2bfloat16(acc[i] * inv_z);
        }
    }
}

}  // namespace

void launch_attention_naive_paged_bf16(
    const void* q,
    const void* k_pages, const void* v_pages,
    void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    int total_tokens,
    int num_requests,
    int num_q_heads, int num_kv_heads,
    int head_dim, int page_size,
    cudaStream_t stream,
    int window_left, float sm_scale)
{
    if (num_requests <= 0 || total_tokens <= 0) return;
    // We launch one block per (request, qo_offset, q_head) — qo_offset
    // is bounded by the largest single-request qo_len. We don't have
    // that bound on hand at the host side, so use `total_tokens` as
    // the conservative upper bound and let the kernel early-exit when
    // `qo_off ≥ qo_hi - qo_lo`. This wastes blocks on small requests
    // but keeps the launch shape uniform.
    dim3 grid(num_requests, total_tokens, num_q_heads);
    dim3 block(BLOCK);
    const std::size_t smem = (head_dim + BLOCK) * sizeof(float);
    naive_paged_attn_bf16_kernel<<<grid, block, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(q),
        static_cast<const __nv_bfloat16*>(k_pages),
        static_cast<const __nv_bfloat16*>(v_pages),
        static_cast<__nv_bfloat16*>(o),
        qo_indptr_d, kv_page_indices_d,
        kv_page_indptr_d, kv_last_page_lens_d,
        num_q_heads, num_kv_heads, head_dim, page_size,
        window_left, sm_scale);
}

void launch_attention_naive_paged(
    const void* q,
    KvCacheLayerView kv_layer,
    void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    int total_tokens,
    int num_requests,
    int num_pages_in_batch,
    int num_q_heads,
    cudaStream_t stream,
    int window_left,
    float sm_scale)
{
    kernels::launch_dequant_kv_cache_layer_to_bf16_active(
        kv_layer, kv_page_indices_d, num_pages_in_batch, stream);
    launch_attention_naive_paged_bf16(
        q,
        kv_layer.k_bf16_pages,
        kv_layer.v_bf16_pages,
        o,
        qo_indptr_d, kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
        total_tokens, num_requests, num_q_heads, kv_layer.num_kv_heads,
        kv_layer.head_dim, kv_layer.page_size, stream, window_left, sm_scale);
}

}  // namespace pie_cuda_driver::ops
