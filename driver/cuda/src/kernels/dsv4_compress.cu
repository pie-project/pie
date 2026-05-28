#include "kernels/dsv4_compress.hpp"

#include <cmath>
#include <cuda_bf16.h>

#include "cuda_check.hpp"

namespace pie_cuda_driver::kernels {

namespace {

constexpr int BLOCK = 256;

__global__ void average_pool_kernel(
    const __nv_bfloat16* __restrict__ input,  // [N, dim]
    __nv_bfloat16* __restrict__ output,       // [N/ratio, dim]
    int N,
    int dim,
    int ratio)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_tokens = N / ratio;
    if (idx >= out_tokens * dim) return;

    const int d = idx % dim;
    const int out_tok = idx / dim;
    const int in_start = out_tok * ratio;

    float sum = 0.f;
    const int end = min(in_start + ratio, N);
    for (int t = in_start; t < end; ++t) {
        sum += __bfloat162float(input[static_cast<long long>(t) * dim + d]);
    }
    output[static_cast<long long>(out_tok) * dim + d] =
        __float2bfloat16(sum / static_cast<float>(end - in_start));
}

__global__ void add_ape_kernel(
    __nv_bfloat16* __restrict__ data,    // [N_compressed, dim]
    const float* __restrict__ ape,       // [ratio, dim]
    int N_compressed,
    int dim,
    int ratio)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_compressed * dim) return;

    const int d = idx % dim;
    const int tok = idx / dim;
    const int pos_in_window = tok % ratio;

    const float val = __bfloat162float(data[idx]) +
                      ape[pos_in_window * dim + d];
    data[idx] = __float2bfloat16(val);
}

// ── Gated softmax pooling kernel ─────────────────────────────────────
// One thread per output element (group_idx, dim_idx).
// For each element: compute softmax over ratio consecutive scores, then
// weighted sum of kv values.
__global__ void gated_softmax_pool_kernel(
    const __nv_bfloat16* __restrict__ kv,      // [N, dim]
    const __nv_bfloat16* __restrict__ score,   // [N, dim]
    __nv_bfloat16* __restrict__ output,        // [N/ratio, dim]
    int N,
    int dim,
    int ratio)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_tokens = N / ratio;
    if (idx >= out_tokens * dim) return;

    const int d = idx % dim;
    const int g = idx / dim;
    const int base = g * ratio;

    // Numerically-stable softmax over ratio elements at dimension d
    float max_s = -INFINITY;
    for (int i = 0; i < ratio && (base + i) < N; ++i) {
        const float s = __bfloat162float(score[static_cast<long long>(base + i) * dim + d]);
        max_s = fmaxf(max_s, s);
    }

    float sum_exp = 0.f;
    float weighted_sum = 0.f;
    for (int i = 0; i < ratio && (base + i) < N; ++i) {
        const long long pos = static_cast<long long>(base + i) * dim + d;
        const float s = __bfloat162float(score[pos]);
        const float v = __bfloat162float(kv[pos]);
        const float e = expf(s - max_s);
        sum_exp += e;
        weighted_sum += v * e;
    }

    output[static_cast<long long>(g) * dim + d] =
        __float2bfloat16(sum_exp > 0.f ? weighted_sum / sum_exp : 0.f);
}

// ── Combine two attention outputs kernel ─────────────────────────────
// One block per (token, head). Threads stride along head_dim.
__global__ void combine_attn_outputs_kernel(
    const __nv_bfloat16* __restrict__ o1,
    const float* __restrict__ lse1,
    const __nv_bfloat16* __restrict__ o2,
    const float* __restrict__ lse2,
    __nv_bfloat16* __restrict__ o_out,
    float* __restrict__ lse_out,
    int num_heads,
    int head_dim)
{
    const int n = blockIdx.x;  // token index
    const int h = blockIdx.y;  // head index

    const float l1 = lse1[n * num_heads + h];
    const float l2 = lse2[n * num_heads + h];

    // If lse2 is -inf, compressed attention had no entries — keep o1 unchanged.
    if (!isfinite(l2)) {
        // Copy o1 to o_out if they differ
        if (o1 != o_out) {
            const long long off = (static_cast<long long>(n) * num_heads + h) * head_dim;
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                o_out[off + d] = o1[off + d];
            }
        }
        if (lse_out != nullptr && lse_out != lse1) {
            if (threadIdx.x == 0) lse_out[n * num_heads + h] = l1;
        }
        return;
    }

    // If lse1 is -inf (SWA had no entries — shouldn't happen but handle), use o2
    if (!isfinite(l1)) {
        const long long off = (static_cast<long long>(n) * num_heads + h) * head_dim;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            o_out[off + d] = o2[off + d];
        }
        if (lse_out != nullptr) {
            if (threadIdx.x == 0) lse_out[n * num_heads + h] = l2;
        }
        return;
    }

    const float lse_max = fmaxf(l1, l2);
    const float w1 = expf(l1 - lse_max);
    const float w2 = expf(l2 - lse_max);
    const float inv_total = 1.0f / (w1 + w2);

    const long long off = (static_cast<long long>(n) * num_heads + h) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v1 = __bfloat162float(o1[off + d]);
        const float v2 = __bfloat162float(o2[off + d]);
        o_out[off + d] = __float2bfloat16((v1 * w1 + v2 * w2) * inv_total);
    }

    if (lse_out != nullptr && threadIdx.x == 0) {
        lse_out[n * num_heads + h] = lse_max + logf(w1 + w2);
    }
}

// ── Dense attention over compressed KV (per-request causal) ──────────
// One block per (request, query_offset, head).
// Each block computes attention for one query row against the compressed KV
// entries of its request, with causal masking: query at local offset t can
// see compressed entry c only if c < (t+1) / ratio.
constexpr int ATTN_BLOCK = 128;
constexpr int ATTN_MAX_HEAD_DIM = ATTN_BLOCK * 8;  // 1024

// Packed parameters uploaded to device memory before kernel launch.
struct CompressedAttnParams {
    int qo_lo;
    int qo_hi;
    int comp_offset;
    int comp_len;
    int comp_ratio;
};

__global__ void compressed_attn_kernel(
    const __nv_bfloat16* __restrict__ q,       // [total_tokens, num_q_heads, head_dim]
    const __nv_bfloat16* __restrict__ comp_kv, // [total_comp, head_dim]
    __nv_bfloat16* __restrict__ o,             // [total_tokens, num_q_heads, head_dim]
    float* __restrict__ lse_out,               // [total_tokens, num_q_heads] or nullptr
    const CompressedAttnParams* __restrict__ params, // [R]
    int num_q_heads,
    int head_dim,
    float scale)
{
    const int r       = blockIdx.x;
    const int qo_off  = blockIdx.y;
    const int q_head  = blockIdx.z;
    const int tid     = threadIdx.x;

    const auto& p = params[r];
    const int qo_lo = p.qo_lo;
    const int qo_hi = p.qo_hi;
    const int comp_off = p.comp_offset;
    const int comp_len = p.comp_len;
    const int ratio = p.comp_ratio;

    if (qo_lo + qo_off >= qo_hi) return;
    const int qi = qo_lo + qo_off;  // absolute query index in the batch

    // How many compressed entries can this query see?
    // Query at local position t (0-indexed within the request) can see
    // compressed entries [0, (t+1)/ratio). The local position is qo_off.
    const int num_visible = min((qo_off + 1) / ratio, comp_len);

    extern __shared__ float smem[];
    float* q_smem = smem;                  // [head_dim]
    float* reduce = smem + head_dim;       // [ATTN_BLOCK]

    // Load query vector into shared memory
    const __nv_bfloat16* q_row =
        q + (static_cast<long long>(qi) * num_q_heads + q_head) * head_dim;
    for (int d = tid; d < head_dim; d += ATTN_BLOCK) {
        q_smem[d] = __bfloat162float(q_row[d]);
    }
    __syncthreads();

    // Output row
    __nv_bfloat16* o_row =
        o + (static_cast<long long>(qi) * num_q_heads + q_head) * head_dim;

    if (num_visible <= 0) {
        // No compressed entries visible — zero output, lse = -inf
        for (int d = tid; d < head_dim; d += ATTN_BLOCK) {
            o_row[d] = __float2bfloat16(0.f);
        }
        if (lse_out != nullptr && tid == 0) {
            lse_out[qi * num_q_heads + q_head] = -INFINITY;
        }
        return;
    }

    // Two-pass attention: find max score, then compute exp-weighted sum
    // Pass 1: find max score
    float local_max = -INFINITY;
    for (int c = tid; c < num_visible; c += ATTN_BLOCK) {
        const __nv_bfloat16* k_row =
            comp_kv + static_cast<long long>(comp_off + c) * head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_smem[d] * __bfloat162float(k_row[d]);
        }
        local_max = fmaxf(local_max, dot * scale);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int off = ATTN_BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) reduce[tid] = fmaxf(reduce[tid], reduce[tid + off]);
        __syncthreads();
    }
    const float row_max = reduce[0];

    // Pass 2: compute exp-weighted sum of V
    const int dims_per_thread = (head_dim + ATTN_BLOCK - 1) / ATTN_BLOCK;
    float acc[8] = {};  // dims_per_thread <= 8 (head_dim <= 1024)
    float local_z = 0.f;

    for (int c = 0; c < num_visible; ++c) {
        // Compute score for this compressed entry
        const __nv_bfloat16* k_row =
            comp_kv + static_cast<long long>(comp_off + c) * head_dim;
        float dot = 0.f;
        for (int d = tid; d < head_dim; d += ATTN_BLOCK) {
            dot += q_smem[d] * __bfloat162float(k_row[d]);
        }
        reduce[tid] = dot;
        __syncthreads();
        for (int off = ATTN_BLOCK / 2; off > 0; off >>= 1) {
            if (tid < off) reduce[tid] += reduce[tid + off];
            __syncthreads();
        }
        const float w = expf(reduce[0] * scale - row_max);
        if (tid == 0) local_z += w;
        __syncthreads();

        // Accumulate V (compressed KV serves as both K and V — MLA style)
        for (int i = 0; i < dims_per_thread; ++i) {
            const int d = tid + i * ATTN_BLOCK;
            if (d < head_dim) {
                acc[i] += w * __bfloat162float(k_row[d]);
            }
        }
    }

    __shared__ float z_shared;
    if (tid == 0) z_shared = local_z;
    __syncthreads();
    const float inv_z = z_shared > 0.f ? 1.0f / z_shared : 0.f;

    if (lse_out != nullptr && tid == 0) {
        lse_out[qi * num_q_heads + q_head] =
            z_shared > 0.f ? (logf(z_shared) + row_max) : -INFINITY;
    }

    for (int i = 0; i < dims_per_thread; ++i) {
        const int d = tid + i * ATTN_BLOCK;
        if (d < head_dim) {
            o_row[d] = __float2bfloat16(acc[i] * inv_z);
        }
    }
}

}  // namespace

void launch_average_pool_bf16(
    const void* input,
    void* output,
    int N,
    int dim,
    int ratio,
    cudaStream_t stream)
{
    const int out_tokens = N / ratio;
    if (out_tokens <= 0 || dim <= 0) return;
    const int total = out_tokens * dim;
    const int grid = (total + BLOCK - 1) / BLOCK;
    average_pool_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<__nv_bfloat16*>(output),
        N, dim, ratio);
}

void launch_add_ape_f32(
    void* data,
    const float* ape,
    int N_compressed,
    int dim,
    int ratio,
    cudaStream_t stream)
{
    if (N_compressed <= 0 || dim <= 0) return;
    const int total = N_compressed * dim;
    const int grid = (total + BLOCK - 1) / BLOCK;
    add_ape_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<__nv_bfloat16*>(data),
        ape,
        N_compressed, dim, ratio);
}

void launch_gated_softmax_pool_bf16(
    const void* kv,
    const void* score,
    void* output,
    int N,
    int dim,
    int ratio,
    cudaStream_t stream)
{
    const int out_tokens = N / ratio;
    if (out_tokens <= 0 || dim <= 0) return;
    const int total = out_tokens * dim;
    const int grid = (total + BLOCK - 1) / BLOCK;
    gated_softmax_pool_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(kv),
        static_cast<const __nv_bfloat16*>(score),
        static_cast<__nv_bfloat16*>(output),
        N, dim, ratio);
}

void launch_combine_attn_outputs_bf16(
    const void* o1, const float* lse1,
    const void* o2, const float* lse2,
    void* o_out, float* lse_out,
    int N, int num_heads, int head_dim,
    cudaStream_t stream)
{
    if (N <= 0) return;
    dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(num_heads));
    const int block = (head_dim < 32) ? 32 : ((head_dim > 256) ? 256 : head_dim);
    combine_attn_outputs_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(o1), lse1,
        static_cast<const __nv_bfloat16*>(o2), lse2,
        static_cast<__nv_bfloat16*>(o_out), lse_out,
        num_heads, head_dim);
}

void launch_attention_compressed_bf16(
    const void* q,
    const void* comp_kv,
    void* o,
    float* lse_out,
    const int* qo_indptr,
    const int* comp_offsets,
    const int* comp_lens,
    const int* comp_ratios,
    int total_tokens,
    int num_requests,
    int num_q_heads,
    int head_dim,
    float sm_scale,
    cudaStream_t stream)
{
    if (num_requests <= 0 || total_tokens <= 0) return;

    // Build params on host, upload to device
    std::vector<CompressedAttnParams> params_h(static_cast<std::size_t>(num_requests));
    for (int r = 0; r < num_requests; ++r) {
        params_h[r].qo_lo = qo_indptr[r];
        params_h[r].qo_hi = qo_indptr[r + 1];
        params_h[r].comp_offset = comp_offsets[r];
        params_h[r].comp_len = comp_lens[r];
        params_h[r].comp_ratio = comp_ratios[r];
    }

    // Allocate device memory for params
    CompressedAttnParams* params_d = nullptr;
    CUDA_CHECK(cudaMallocAsync(&params_d,
        sizeof(CompressedAttnParams) * num_requests, stream));
    CUDA_CHECK(cudaMemcpyAsync(params_d, params_h.data(),
        sizeof(CompressedAttnParams) * num_requests,
        cudaMemcpyHostToDevice, stream));

    dim3 grid(num_requests, total_tokens, num_q_heads);
    dim3 block(ATTN_BLOCK);
    const std::size_t smem = (static_cast<std::size_t>(head_dim) + ATTN_BLOCK) * sizeof(float);

    compressed_attn_kernel<<<grid, block, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(q),
        static_cast<const __nv_bfloat16*>(comp_kv),
        static_cast<__nv_bfloat16*>(o),
        lse_out,
        params_d,
        num_q_heads, head_dim, sm_scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(params_d, stream));
}

}  // namespace pie_cuda_driver::kernels
