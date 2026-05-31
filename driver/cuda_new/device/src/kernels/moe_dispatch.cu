#include "moe_dispatch.cuh"

#include <cuda_bf16.h>

// Lifted from driver/cuda/src/kernels/moe_dispatch.cu. The scatter/gather/
// permute/combine kernels and their anon-namespace helpers are copied
// VERBATIM; the only changes are the namespace (pie_cuda_driver ->
// pie_cuda_device) and dropping the `launch_` prefix on the entry points.
//
// The cuBLAS pointer-array builders (build_moe_ptrs_*, build_dual_bf16_gemm_ptrs)
// and the WMMA per-expert GEMM kernels (moe_*_decode_wmma_bf16, which pulled in
// <mma.h>) are NOT lifted — they are the grouped-GEMM plumbing / the GEMM
// itself, not the routing dispatch core. With them gone, <mma.h> is no longer
// needed here.

namespace pie_cuda_device::kernels {

namespace {

constexpr int BLOCK = 256;

__global__ void scatter_add_weighted_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ src,
    const std::int32_t* __restrict__ dst_idx,
    const float* __restrict__ row_weights,
    int hidden)
{
    const int n = blockIdx.x;
    const int row = dst_idx[n];
    const float w = row_weights[n];
    const __nv_bfloat16* in = src + static_cast<long long>(n) * hidden;
    __nv_bfloat16*       o  = out + static_cast<long long>(row) * hidden;
    for (int h = threadIdx.x; h < hidden; h += BLOCK) {
        const float prev = __bfloat162float(o[h]);
        const float add  = __bfloat162float(in[h]) * w;
        o[h] = __float2bfloat16(prev + add);
    }
}

}  // namespace

void scatter_add_weighted_bf16(
    void* out, const void* src,
    const std::int32_t* dst_idx, const float* row_weights,
    int num_routed, int hidden, cudaStream_t stream)
{
    if (num_routed <= 0) return;
    scatter_add_weighted_bf16_kernel<<<num_routed, BLOCK, 0, stream>>>(
        static_cast<__nv_bfloat16*>(out),
        static_cast<const __nv_bfloat16*>(src),
        dst_idx, row_weights,
        hidden);
}

namespace {

__global__ void scalar_weighted_add_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ src,
    float weight, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float ov = __bfloat162float(out[i]);
    const float sv = __bfloat162float(src[i]);
    out[i] = __float2bfloat16(ov + weight * sv);
}

}  // namespace

void scalar_weighted_add_bf16(
    void* out, const void* src, float weight, int n, cudaStream_t stream)
{
    if (n <= 0) return;
    constexpr int BS = 256;
    const int grid = (n + BS - 1) / BS;
    scalar_weighted_add_bf16_kernel<<<grid, BS, 0, stream>>>(
        static_cast<__nv_bfloat16*>(out),
        static_cast<const __nv_bfloat16*>(src),
        weight, n);
}

namespace {

__global__ void batched_weighted_sum_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ src,   // [batch, hidden]
    const float* __restrict__ weights,       // [batch]
    int batch, int hidden)
{
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= hidden) return;
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < 16; ++k) {  // unroll up to top_k=16; loop bounded
        if (k >= batch) break;
        const float v = __bfloat162float(src[(long long)k * hidden + h]);
        acc += weights[k] * v;
    }
    out[h] = __float2bfloat16(acc);
}

}  // namespace

void batched_weighted_sum_bf16(
    void* out, const void* src, const float* weights,
    int batch, int hidden, cudaStream_t stream)
{
    if (batch <= 0 || hidden <= 0) return;
    constexpr int BS = 256;
    const int grid = (hidden + BS - 1) / BS;
    batched_weighted_sum_bf16_kernel<<<grid, BS, 0, stream>>>(
        static_cast<__nv_bfloat16*>(out),
        static_cast<const __nv_bfloat16*>(src),
        weights, batch, hidden);
}

namespace {

__global__ void token_batched_weighted_sum_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ src,   // [num_tokens, top_k, hidden]
    const float* __restrict__ weights,       // [num_tokens, top_k]
    int top_k, int hidden)
{
    const int n = blockIdx.y;
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= hidden) return;
    const long long base = static_cast<long long>(n) * top_k;
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        if (k >= top_k) break;
        const long long r = base + k;
        const float v = __bfloat162float(src[r * hidden + h]);
        acc += weights[r] * v;
    }
    out[static_cast<long long>(n) * hidden + h] = __float2bfloat16(acc);
}

__global__ void token_batched_weighted_sum_add_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ src,   // [num_tokens, top_k, hidden]
    const float* __restrict__ weights,       // [num_tokens, top_k]
    int top_k, int hidden)
{
    const int n = blockIdx.y;
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= hidden) return;
    const long long base = static_cast<long long>(n) * top_k;
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        if (k >= top_k) break;
        const long long r = base + k;
        const float v = __bfloat162float(src[r * hidden + h]);
        acc += weights[r] * v;
    }
    const long long out_idx = static_cast<long long>(n) * hidden + h;
    out[out_idx] = __float2bfloat16(__bfloat162float(out[out_idx]) + acc);
}

}  // namespace

void token_batched_weighted_sum_bf16(
    void* out, const void* src, const float* weights,
    int num_tokens, int top_k, int hidden, cudaStream_t stream)
{
    if (num_tokens <= 0 || top_k <= 0 || hidden <= 0) return;
    constexpr int BS = 256;
    const dim3 grid((hidden + BS - 1) / BS, num_tokens);
    token_batched_weighted_sum_bf16_kernel<<<grid, BS, 0, stream>>>(
        static_cast<__nv_bfloat16*>(out),
        static_cast<const __nv_bfloat16*>(src),
        weights, top_k, hidden);
}

void token_batched_weighted_sum_add_bf16(
    void* out, const void* src, const float* weights,
    int num_tokens, int top_k, int hidden, cudaStream_t stream)
{
    if (num_tokens <= 0 || top_k <= 0 || hidden <= 0) return;
    constexpr int BS = 256;
    const dim3 grid((hidden + BS - 1) / BS, num_tokens);
    token_batched_weighted_sum_add_bf16_kernel<<<grid, BS, 0, stream>>>(
        static_cast<__nv_bfloat16*>(out),
        static_cast<const __nv_bfloat16*>(src),
        weights, top_k, hidden);
}

namespace {

__global__ void token_batched_weighted_sum_aligned_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ aligned_out,
    const float* __restrict__ weights,
    const std::int32_t* __restrict__ route_to_aligned_row,
    int top_k,
    int hidden)
{
    const int n = blockIdx.y;
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= hidden) return;
    const long long base = static_cast<long long>(n) * top_k;
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        if (k >= top_k) break;
        const long long route = base + k;
        const int row = route_to_aligned_row[route];
        const float v = __bfloat162float(
            aligned_out[static_cast<long long>(row) * hidden + h]);
        acc += weights[route] * v;
    }
    out[static_cast<long long>(n) * hidden + h] = __float2bfloat16(acc);
}

}  // namespace

void token_batched_weighted_sum_aligned_bf16(
    void* out,
    const void* aligned_out,
    const float* weights,
    const std::int32_t* route_to_aligned_row,
    int num_tokens,
    int top_k,
    int hidden,
    cudaStream_t stream)
{
    if (num_tokens <= 0 || top_k <= 0 || hidden <= 0) return;
    constexpr int BS = 256;
    const dim3 grid((hidden + BS - 1) / BS, num_tokens);
    token_batched_weighted_sum_aligned_bf16_kernel<<<grid, BS, 0, stream>>>(
        static_cast<__nv_bfloat16*>(out),
        static_cast<const __nv_bfloat16*>(aligned_out),
        weights,
        route_to_aligned_row,
        top_k,
        hidden);
}

namespace {

__global__ void moe_align_decode_kernel(
    const std::int32_t* __restrict__ topk_idx,
    std::int32_t* __restrict__ sorted_route_ids,
    std::int32_t* __restrict__ expert_ids,
    std::int32_t* __restrict__ route_to_aligned_row,
    int num_routes,
    int num_experts,
    int block_size,
    int max_blocks)
{
    extern __shared__ std::int32_t align_smem[];
    std::int32_t* counts = align_smem;
    std::int32_t* offsets = counts + num_experts;
    std::int32_t* fill = offsets + num_experts + 1;

    const int aligned_rows = max_blocks * block_size;
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        counts[i] = 0;
        fill[i] = 0;
    }
    for (int i = threadIdx.x; i < aligned_rows; i += blockDim.x) {
        sorted_route_ids[i] = num_routes;
    }
    for (int i = threadIdx.x; i < max_blocks; i += blockDim.x) {
        expert_ids[i] = -1;
    }
    __syncthreads();

    for (int r = threadIdx.x; r < num_routes; r += blockDim.x) {
        const int e = topk_idx[r];
        if (0 <= e && e < num_experts) {
            atomicAdd(counts + e, 1);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int running = 0;
        for (int e = 0; e < num_experts; ++e) {
            offsets[e] = running;
            const int c = counts[e];
            running += ((c + block_size - 1) / block_size) * block_size;
        }
        offsets[num_experts] = running;
    }
    __syncthreads();

    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        const int begin = offsets[e];
        const int end = offsets[e + 1];
        for (int row = begin; row < end; row += block_size) {
            const int b = row / block_size;
            if (b < max_blocks) expert_ids[b] = e;
        }
    }
    __syncthreads();

    for (int r = threadIdx.x; r < num_routes; r += blockDim.x) {
        const int e = topk_idx[r];
        if (0 <= e && e < num_experts) {
            const int pos = atomicAdd(fill + e, 1);
            const int out = offsets[e] + pos;
            if (out < aligned_rows) {
                sorted_route_ids[out] = r;
                if (route_to_aligned_row != nullptr) {
                    route_to_aligned_row[r] = out;
                }
            }
        }
    }
}

__global__ void moe_bucket_exact_kernel(
    const std::int32_t* __restrict__ topk_idx,
    std::int32_t* __restrict__ sorted_route_ids,
    std::int32_t* __restrict__ route_to_sorted_row,
    std::int32_t* __restrict__ counts_out,
    int num_routes,
    int num_experts)
{
    extern __shared__ std::int32_t bucket_smem[];
    std::int32_t* counts = bucket_smem;
    std::int32_t* offsets = counts + num_experts;
    std::int32_t* fill = offsets + num_experts + 1;

    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        counts[i] = 0;
        fill[i] = 0;
    }
    __syncthreads();

    for (int r = threadIdx.x; r < num_routes; r += blockDim.x) {
        const int e = topk_idx[r];
        if (0 <= e && e < num_experts) {
            atomicAdd(counts + e, 1);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int running = 0;
        for (int e = 0; e < num_experts; ++e) {
            offsets[e] = running;
            const int c = counts[e];
            counts_out[e] = c;
            running += c;
        }
        offsets[num_experts] = running;
    }
    __syncthreads();

    for (int r = threadIdx.x; r < num_routes; r += blockDim.x) {
        const int e = topk_idx[r];
        if (0 <= e && e < num_experts) {
            const int pos = atomicAdd(fill + e, 1);
            const int out = offsets[e] + pos;
            sorted_route_ids[out] = r;
            route_to_sorted_row[r] = out;
        }
    }
}

__global__ void gather_moe_aligned_inputs_bf16_kernel(
    const __nv_bfloat16* __restrict__ norm_x,
    const std::int32_t* __restrict__ sorted_route_ids,
    __nv_bfloat16* __restrict__ aligned_in,
    int num_routes,
    int aligned_rows,
    int top_k,
    int hidden)
{
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (h >= hidden || row >= aligned_rows) return;
    const int route = sorted_route_ids[row];
    __nv_bfloat16 v = __float2bfloat16(0.0f);
    if (route < num_routes) {
        const int token = route / top_k;
        v = norm_x[static_cast<long long>(token) * hidden + h];
    }
    aligned_in[static_cast<long long>(row) * hidden + h] = v;
}

__global__ void reorder_moe_aligned_output_bf16_kernel(
    const __nv_bfloat16* __restrict__ aligned_out,
    const std::int32_t* __restrict__ sorted_route_ids,
    __nv_bfloat16* __restrict__ route_out,
    int num_routes,
    int aligned_rows,
    int hidden)
{
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (h >= hidden || row >= aligned_rows) return;
    const int route = sorted_route_ids[row];
    if (route >= num_routes) return;
    route_out[static_cast<long long>(route) * hidden + h] =
        aligned_out[static_cast<long long>(row) * hidden + h];
}

}  // namespace

void moe_align_decode(
    const std::int32_t* topk_idx,
    std::int32_t* sorted_route_ids,
    std::int32_t* expert_ids,
    std::int32_t* route_to_aligned_row,
    int num_routes,
    int num_experts,
    int block_size,
    int max_blocks,
    cudaStream_t stream)
{
    if (num_routes <= 0 || num_experts <= 0 || block_size <= 0 ||
        max_blocks <= 0) {
        return;
    }
    constexpr int BS = 1024;
    const std::size_t smem =
        static_cast<std::size_t>(3 * num_experts + 1) * sizeof(std::int32_t);
    moe_align_decode_kernel<<<1, BS, smem, stream>>>(
        topk_idx, sorted_route_ids, expert_ids, route_to_aligned_row,
        num_routes, num_experts, block_size, max_blocks);
}

void moe_bucket_exact(
    const std::int32_t* topk_idx,
    std::int32_t* sorted_route_ids,
    std::int32_t* route_to_sorted_row,
    std::int32_t* counts_out,
    int num_routes,
    int num_experts,
    cudaStream_t stream)
{
    if (num_routes <= 0 || num_experts <= 0) return;
    constexpr int BS = 1024;
    const std::size_t smem =
        static_cast<std::size_t>(3 * num_experts + 1) * sizeof(std::int32_t);
    moe_bucket_exact_kernel<<<1, BS, smem, stream>>>(
        topk_idx, sorted_route_ids, route_to_sorted_row, counts_out,
        num_routes, num_experts);
}

void gather_moe_aligned_inputs_bf16(
    const void* norm_x,
    const std::int32_t* sorted_route_ids,
    void* aligned_in,
    int num_routes,
    int aligned_rows,
    int top_k,
    int hidden,
    cudaStream_t stream)
{
    if (aligned_rows <= 0 || hidden <= 0) return;
    constexpr int BS = 256;
    const dim3 grid((hidden + BS - 1) / BS, aligned_rows);
    gather_moe_aligned_inputs_bf16_kernel<<<grid, BS, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(norm_x),
        sorted_route_ids,
        static_cast<__nv_bfloat16*>(aligned_in),
        num_routes, aligned_rows, top_k, hidden);
}

void reorder_moe_aligned_output_bf16(
    const void* aligned_out,
    const std::int32_t* sorted_route_ids,
    void* route_out,
    int num_routes,
    int aligned_rows,
    int hidden,
    cudaStream_t stream)
{
    if (aligned_rows <= 0 || hidden <= 0) return;
    constexpr int BS = 256;
    const dim3 grid((hidden + BS - 1) / BS, aligned_rows);
    reorder_moe_aligned_output_bf16_kernel<<<grid, BS, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(aligned_out),
        sorted_route_ids,
        static_cast<__nv_bfloat16*>(route_out),
        num_routes, aligned_rows, hidden);
}

}  // namespace pie_cuda_device::kernels
