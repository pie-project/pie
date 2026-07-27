#include "kernels/moe_dispatch.hpp"

#include <cstdint>
#include <cuda_bf16.h>
#include <mma.h>

namespace pie_cuda_driver::kernels {

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

void launch_scatter_add_weighted_bf16(
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

void launch_scalar_weighted_add_bf16(
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

__global__ void build_dual_bf16_gemm_ptrs_kernel(
    const __nv_bfloat16* act,
    const __nv_bfloat16* w0,
    const __nv_bfloat16* w1,
    __nv_bfloat16* out0,
    __nv_bfloat16* out1,
    const __nv_bfloat16** act_ptrs,
    const __nv_bfloat16** w_ptrs,
    __nv_bfloat16** out_ptrs)
{
    act_ptrs[0] = act;
    act_ptrs[1] = act;
    w_ptrs[0] = w0;
    w_ptrs[1] = w1;
    out_ptrs[0] = out0;
    out_ptrs[1] = out1;
}

}  // namespace

void launch_build_dual_bf16_gemm_ptrs(
    const void* act,
    const void* w0,
    const void* w1,
    void* out0,
    void* out1,
    const void** act_ptrs,
    const void** w_ptrs,
    void** out_ptrs,
    cudaStream_t stream)
{
    build_dual_bf16_gemm_ptrs_kernel<<<1, 1, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(act),
        static_cast<const __nv_bfloat16*>(w0),
        static_cast<const __nv_bfloat16*>(w1),
        static_cast<__nv_bfloat16*>(out0),
        static_cast<__nv_bfloat16*>(out1),
        reinterpret_cast<const __nv_bfloat16**>(act_ptrs),
        reinterpret_cast<const __nv_bfloat16**>(w_ptrs),
        reinterpret_cast<__nv_bfloat16**>(out_ptrs));
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

void launch_batched_weighted_sum_bf16(
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

void launch_token_batched_weighted_sum_bf16(
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

void launch_token_batched_weighted_sum_add_bf16(
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

void launch_token_batched_weighted_sum_aligned_bf16(
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

// One block, top_k threads. Each thread is responsible for one of the
// active experts: looks up the expert ID and emits a row of pointers
// into the cuBLAS batched-gemm input arrays.
__global__ void build_moe_ptrs_decode_bf16_kernel(
    const std::int32_t* topk_idx,
    const float*        topk_w,
    const __nv_bfloat16* gate_up_base,
    const __nv_bfloat16* down_base,
    const __nv_bfloat16* norm_x,
    __nv_bfloat16* expert_gate_up,
    __nv_bfloat16* expert_act,
    __nv_bfloat16* expert_out,
    const __nv_bfloat16** a_gu_ptrs,
    const __nv_bfloat16** b_gu_ptrs,
    __nv_bfloat16**       c_gu_ptrs,
    const __nv_bfloat16** a_dn_ptrs,
    const __nv_bfloat16** b_dn_ptrs,
    __nv_bfloat16**       c_dn_ptrs,
    float*                weights_out,
    int top_k, int H, int I_moe)
{
    const int k = threadIdx.x;
    if (k >= top_k) return;
    const long long stride_gu = 2LL * I_moe * H;
    const long long stride_dn = (long long)H * I_moe;
    const int e = topk_idx[k];

    a_gu_ptrs[k] = gate_up_base + e * stride_gu;
    b_gu_ptrs[k] = norm_x;
    c_gu_ptrs[k] = expert_gate_up + (long long)k * 2 * I_moe;

    a_dn_ptrs[k] = down_base + e * stride_dn;
    b_dn_ptrs[k] = expert_act + (long long)k * I_moe;
    c_dn_ptrs[k] = expert_out + (long long)k * H;

    weights_out[k] = topk_w[k];
}

void launch_build_moe_ptrs_decode_bf16(
    const std::int32_t* topk_idx,
    const float*        topk_w,
    const void* gate_up_base, const void* down_base, const void* norm_x,
    void* expert_gate_up, void* expert_act, void* expert_out,
    const void** a_gu_ptrs, const void** b_gu_ptrs, void** c_gu_ptrs,
    const void** a_dn_ptrs, const void** b_dn_ptrs, void** c_dn_ptrs,
    float*       weights_out,
    int top_k, int H, int I_moe, cudaStream_t stream)
{
    if (top_k <= 0) return;
    build_moe_ptrs_decode_bf16_kernel<<<1, top_k, 0, stream>>>(
        topk_idx, topk_w,
        static_cast<const __nv_bfloat16*>(gate_up_base),
        static_cast<const __nv_bfloat16*>(down_base),
        static_cast<const __nv_bfloat16*>(norm_x),
        static_cast<__nv_bfloat16*>(expert_gate_up),
        static_cast<__nv_bfloat16*>(expert_act),
        static_cast<__nv_bfloat16*>(expert_out),
        reinterpret_cast<const __nv_bfloat16**>(a_gu_ptrs),
        reinterpret_cast<const __nv_bfloat16**>(b_gu_ptrs),
        reinterpret_cast<__nv_bfloat16**>(c_gu_ptrs),
        reinterpret_cast<const __nv_bfloat16**>(a_dn_ptrs),
        reinterpret_cast<const __nv_bfloat16**>(b_dn_ptrs),
        reinterpret_cast<__nv_bfloat16**>(c_dn_ptrs),
        weights_out,
        top_k, H, I_moe);
}

namespace {

__global__ void build_moe_ptrs_decode_batched_bf16_kernel(
    const std::int32_t* topk_idx,
    const float*        topk_w,
    const __nv_bfloat16* gate_up_base,
    const __nv_bfloat16* down_base,
    const __nv_bfloat16* norm_x,
    __nv_bfloat16* expert_gate_up,
    __nv_bfloat16* expert_act,
    __nv_bfloat16* expert_out,
    const __nv_bfloat16** a_gu_ptrs,
    const __nv_bfloat16** b_gu_ptrs,
    __nv_bfloat16**       c_gu_ptrs,
    const __nv_bfloat16** a_dn_ptrs,
    const __nv_bfloat16** b_dn_ptrs,
    __nv_bfloat16**       c_dn_ptrs,
    float*                weights_out,
    int num_tokens, int top_k, int H, int I_moe)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_tokens * top_k;
    if (r >= total) return;
    const long long stride_gu = 2LL * I_moe * H;
    const long long stride_dn = static_cast<long long>(H) * I_moe;
    const int token = r / top_k;
    const int e = topk_idx[r];

    a_gu_ptrs[r] = gate_up_base + static_cast<long long>(e) * stride_gu;
    b_gu_ptrs[r] = norm_x + static_cast<long long>(token) * H;
    c_gu_ptrs[r] = expert_gate_up + static_cast<long long>(r) * 2 * I_moe;

    a_dn_ptrs[r] = down_base + static_cast<long long>(e) * stride_dn;
    b_dn_ptrs[r] = expert_act + static_cast<long long>(r) * I_moe;
    c_dn_ptrs[r] = expert_out + static_cast<long long>(r) * H;

    weights_out[r] = topk_w[r];
}

}  // namespace

void launch_build_moe_ptrs_decode_batched_bf16(
    const std::int32_t* topk_idx,
    const float*        topk_w,
    const void* gate_up_base, const void* down_base, const void* norm_x,
    void* expert_gate_up, void* expert_act, void* expert_out,
    const void** a_gu_ptrs, const void** b_gu_ptrs, void** c_gu_ptrs,
    const void** a_dn_ptrs, const void** b_dn_ptrs, void** c_dn_ptrs,
    float*       weights_out,
    int num_tokens, int top_k, int H, int I_moe, cudaStream_t stream)
{
    const int total = num_tokens * top_k;
    if (total <= 0) return;
    constexpr int BS = 256;
    const int grid = (total + BS - 1) / BS;
    build_moe_ptrs_decode_batched_bf16_kernel<<<grid, BS, 0, stream>>>(
        topk_idx, topk_w,
        static_cast<const __nv_bfloat16*>(gate_up_base),
        static_cast<const __nv_bfloat16*>(down_base),
        static_cast<const __nv_bfloat16*>(norm_x),
        static_cast<__nv_bfloat16*>(expert_gate_up),
        static_cast<__nv_bfloat16*>(expert_act),
        static_cast<__nv_bfloat16*>(expert_out),
        reinterpret_cast<const __nv_bfloat16**>(a_gu_ptrs),
        reinterpret_cast<const __nv_bfloat16**>(b_gu_ptrs),
        reinterpret_cast<__nv_bfloat16**>(c_gu_ptrs),
        reinterpret_cast<const __nv_bfloat16**>(a_dn_ptrs),
        reinterpret_cast<const __nv_bfloat16**>(b_dn_ptrs),
        reinterpret_cast<__nv_bfloat16**>(c_dn_ptrs),
        weights_out,
        num_tokens, top_k, H, I_moe);
}

namespace {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
using namespace nvcuda;
#endif

template <bool ActByToken>
__global__ void moe_decode_wmma_bf16_kernel(
    const std::int32_t* __restrict__ topk_idx,
    const __nv_bfloat16* __restrict__ act,
    const __nv_bfloat16* __restrict__ weight_base,
    __nv_bfloat16* __restrict__ out,
    int top_k,
    int K,
    int N,
    long long expert_stride)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int N_TILE = 64;
    const int n0 = blockIdx.x * N_TILE;
    const int route = blockIdx.y;
    const int expert = topk_idx[route];
    if (expert < 0 || n0 >= N) return;

    extern __shared__ __align__(16) unsigned char wmma_smem[];
    auto* a_tile = reinterpret_cast<__nv_bfloat16*>(wmma_smem);
    auto* c_tile = reinterpret_cast<float*>(a_tile + 16 * 16);
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int n_warp = n0 + warp_id * 16;

    const int token = route / top_k;
    const __nv_bfloat16* act_row =
        act + static_cast<long long>(ActByToken ? token : route) * K;
    const __nv_bfloat16* weight =
        weight_base + static_cast<long long>(expert) * expert_stride;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int i = threadIdx.x; i < 16 * 16; i += blockDim.x) {
            a_tile[i] = __float2bfloat16(0.0f);
        }
        if (threadIdx.x < 16) {
            a_tile[threadIdx.x] = act_row[k0 + threadIdx.x];
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag, a_tile, 16);
        wmma::load_matrix_sync(
            b_frag,
            weight + static_cast<long long>(n_warp) * K + k0,
            K);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncthreads();
    }

    wmma::store_matrix_sync(
        c_tile + warp_id * 16 * 16, acc_frag, 16, wmma::mem_row_major);
    __syncthreads();

    if (lane < 16) {
        const long long out_base = static_cast<long long>(route) * N + n0;
        out[out_base + warp_id * 16 + lane] =
            __float2bfloat16(c_tile[warp_id * 16 * 16 + lane]);
    }
#endif
}

// Bandwidth-optimal decode GEMV for the sparse MoE hot path.
//
// At decode the routed GEMMs are M=1: 8 experts x [1, H] x [H, N]. There
// is no reuse of the weight to exploit, so the only thing that matters is
// reading it at full HBM bandwidth. Tensor cores buy nothing at M=1 (the
// WMMA path measured 45% SLOWER end-to-end), and `cublasGemmBatchedEx`
// leaves bandwidth on the table because its tiling is chosen for shapes
// with an M to fill. One warp per output row, float4 loads, fp32
// accumulate, one shuffle reduction: 1624 GB/s vs cuBLAS's 1282 on A100
// for Qwen3.6's gate/up shape (routes=8, N=1024, K=2048).
//
// `ActByToken` selects the input row: gate/up reads the token's hidden
// state (shared by the token's top_k routes), down reads the route's own
// activation.
template <bool ActByToken, int kWarps>
__global__ void moe_decode_gemv_bf16_kernel(
    const std::int32_t* __restrict__ topk_idx,
    const __nv_bfloat16* __restrict__ act,
    const __nv_bfloat16* __restrict__ weight_base,
    __nv_bfloat16* __restrict__ out,
    int top_k, int K, int N, long long expert_stride)
{
    const int route = blockIdx.y;
    const int row = blockIdx.x * kWarps + threadIdx.y;
    if (row >= N) return;
    const int expert = topk_idx[route];
    const __nv_bfloat16* w =
        weight_base + expert * expert_stride + (long long)row * K;
    const __nv_bfloat16* x =
        act + (long long)(ActByToken ? route / top_k : route) * K;

    const int lane = threadIdx.x;
    float acc = 0.f;
    const int vec = K / 8;
    const float4* w4 = reinterpret_cast<const float4*>(w);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (int i = lane; i < vec; i += 32) {
        float4 wv = w4[i];
        float4 xv = x4[i];
        const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);
        const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&xv);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc += __bfloat162float(wb[j]) * __bfloat162float(xb[j]);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) out[(long long)route * N + row] = __float2bfloat16(acc);
}

}  // namespace

void launch_moe_gate_up_decode_gemv_bf16(
    const std::int32_t* topk_idx,
    const void* norm_x,
    const void* gate_up_base,
    void* expert_gate_up,
    int num_tokens,
    int top_k,
    int H,
    int I_moe,
    cudaStream_t stream)
{
    const int routes = num_tokens * top_k;
    const int N = 2 * I_moe;
    // float4 loads need every row to start 16-byte aligned, which holds
    // iff the reduction extent is a multiple of 8 bf16.
    if (routes <= 0 || H <= 0 || N <= 0 || (H % 8) != 0) return;
    constexpr int kWarps = 4;
    const dim3 grid((N + kWarps - 1) / kWarps, routes);
    const dim3 block(32, kWarps);
    moe_decode_gemv_bf16_kernel</*ActByToken=*/true, kWarps>
        <<<grid, block, 0, stream>>>(
            topk_idx,
            static_cast<const __nv_bfloat16*>(norm_x),
            static_cast<const __nv_bfloat16*>(gate_up_base),
            static_cast<__nv_bfloat16*>(expert_gate_up),
            top_k, H, N, static_cast<long long>(N) * H);
}

void launch_moe_down_decode_gemv_bf16(
    const std::int32_t* topk_idx,
    const void* expert_act,
    const void* down_base,
    void* expert_out,
    int num_tokens,
    int top_k,
    int H,
    int I_moe,
    cudaStream_t stream)
{
    const int routes = num_tokens * top_k;
    if (routes <= 0 || H <= 0 || I_moe <= 0 || (I_moe % 8) != 0) return;
    constexpr int kWarps = 4;
    const dim3 grid((H + kWarps - 1) / kWarps, routes);
    const dim3 block(32, kWarps);
    moe_decode_gemv_bf16_kernel</*ActByToken=*/false, kWarps>
        <<<grid, block, 0, stream>>>(
            topk_idx,
            static_cast<const __nv_bfloat16*>(expert_act),
            static_cast<const __nv_bfloat16*>(down_base),
            static_cast<__nv_bfloat16*>(expert_out),
            top_k, I_moe, H, static_cast<long long>(H) * I_moe);
}

void launch_moe_gate_up_decode_wmma_bf16(
    const std::int32_t* topk_idx,
    const void* norm_x,
    const void* gate_up_base,
    void* expert_gate_up,
    int num_tokens,
    int top_k,
    int H,
    int I_moe,
    cudaStream_t stream)
{
    const int routes = num_tokens * top_k;
    const int N = 2 * I_moe;
    if (routes <= 0 || H <= 0 || N <= 0 || (H % 16) != 0 || (N % 64) != 0) {
        return;
    }
    const dim3 grid(N / 64, routes);
    const std::size_t smem =
        (16 * 16 * sizeof(__nv_bfloat16)) + (4 * 16 * 16 * sizeof(float));
    moe_decode_wmma_bf16_kernel</*ActByToken=*/true><<<grid, 128, smem, stream>>>(
        topk_idx,
        static_cast<const __nv_bfloat16*>(norm_x),
        static_cast<const __nv_bfloat16*>(gate_up_base),
        static_cast<__nv_bfloat16*>(expert_gate_up),
        top_k, H, N, static_cast<long long>(N) * H);
}

void launch_moe_down_decode_wmma_bf16(
    const std::int32_t* topk_idx,
    const void* expert_act,
    const void* down_base,
    void* expert_out,
    int num_tokens,
    int top_k,
    int H,
    int I_moe,
    cudaStream_t stream)
{
    const int routes = num_tokens * top_k;
    if (routes <= 0 || H <= 0 || I_moe <= 0 ||
        (I_moe % 16) != 0 || (H % 64) != 0) {
        return;
    }
    const dim3 grid(H / 64, routes);
    const std::size_t smem =
        (16 * 16 * sizeof(__nv_bfloat16)) + (4 * 16 * 16 * sizeof(float));
    moe_decode_wmma_bf16_kernel</*ActByToken=*/false><<<grid, 128, smem, stream>>>(
        topk_idx,
        static_cast<const __nv_bfloat16*>(expert_act),
        static_cast<const __nv_bfloat16*>(down_base),
        static_cast<__nv_bfloat16*>(expert_out),
        top_k, I_moe, H, static_cast<long long>(H) * I_moe);
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
    std::int32_t* warp_totals = fill + num_experts;
    std::int32_t* block_base = warp_totals + 32;

    const int aligned_rows = max_blocks * block_size;
    if (threadIdx.x == 0) *block_base = 0;
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

    // Block-wide exclusive scan of the per-expert padded block counts.
    // Running this serially on thread 0 cost ~15 us per layer -- 256
    // dependent shared-memory reads and integer divides, x40 layers.
    {
        const int lane = threadIdx.x & 31;
        const int warp = static_cast<int>(threadIdx.x) >> 5;
        const int num_warps = static_cast<int>(blockDim.x) >> 5;
        for (int base = 0; base < num_experts; base += static_cast<int>(blockDim.x)) {
            const int e = base + static_cast<int>(threadIdx.x);
            int padded = 0;
            if (e < num_experts) {
                const int c = counts[e];
                padded = ((c + block_size - 1) / block_size) * block_size;
            }
            int value = padded;
            for (int off = 1; off < 32; off <<= 1) {
                const int n = __shfl_up_sync(0xffffffffu, value, off);
                if (lane >= off) value += n;
            }
            if (lane == 31) warp_totals[warp] = value;
            __syncthreads();
            if (warp == 0) {
                int t = (lane < num_warps) ? warp_totals[lane] : 0;
                for (int off = 1; off < 32; off <<= 1) {
                    const int n = __shfl_up_sync(0xffffffffu, t, off);
                    if (lane >= off) t += n;
                }
                if (lane < num_warps) warp_totals[lane] = t;
            }
            __syncthreads();
            const int warp_prefix = (warp == 0) ? 0 : warp_totals[warp - 1];
            if (e < num_experts) {
                // Exclusive within this chunk, plus everything before it.
                offsets[e] = *block_base + warp_prefix + value - padded;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                *block_base += warp_totals[num_warps - 1];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) offsets[num_experts] = *block_base;
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

// One bf16 per thread makes every warp issue a 64-byte store, half of a
// cache line. Moving 8 (a uint4) per thread turns that into a full 512-byte
// contiguous store per warp.
constexpr int kMoeVecWidth = 8;

__global__ void gather_moe_aligned_inputs_bf16_vec_kernel(
    const __nv_bfloat16* __restrict__ norm_x,
    const std::int32_t* __restrict__ sorted_route_ids,
    __nv_bfloat16* __restrict__ aligned_in,
    int num_routes,
    int aligned_rows,
    int top_k,
    int hidden_vec)
{
    const int hv = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (hv >= hidden_vec || row >= aligned_rows) return;
    const int route = sorted_route_ids[row];
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (route < num_routes) {
        const int token = route / top_k;
        v = reinterpret_cast<const uint4*>(norm_x)[
            static_cast<long long>(token) * hidden_vec + hv];
    }
    reinterpret_cast<uint4*>(aligned_in)[
        static_cast<long long>(row) * hidden_vec + hv] = v;
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

__global__ void build_moe_ptrs_aligned_bf16_kernel(
    const std::int32_t* __restrict__ expert_ids,
    const __nv_bfloat16* __restrict__ gate_up_base,
    const __nv_bfloat16* __restrict__ down_base,
    const __nv_bfloat16* __restrict__ aligned_in,
    __nv_bfloat16* __restrict__ aligned_gate_up,
    __nv_bfloat16* __restrict__ aligned_act,
    __nv_bfloat16* __restrict__ aligned_out,
    const __nv_bfloat16** __restrict__ a_gu_ptrs,
    const __nv_bfloat16** __restrict__ b_gu_ptrs,
    __nv_bfloat16** __restrict__ c_gu_ptrs,
    const __nv_bfloat16** __restrict__ a_dn_ptrs,
    const __nv_bfloat16** __restrict__ b_dn_ptrs,
    __nv_bfloat16** __restrict__ c_dn_ptrs,
    int max_blocks,
    int block_size,
    int H,
    int I_moe)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= max_blocks) return;
    int e = expert_ids[b];
    if (e < 0) e = 0;
    const long long row = static_cast<long long>(b) * block_size;
    const long long stride_gu = 2LL * I_moe * H;
    const long long stride_dn = static_cast<long long>(H) * I_moe;

    a_gu_ptrs[b] = gate_up_base + static_cast<long long>(e) * stride_gu;
    b_gu_ptrs[b] = aligned_in + row * H;
    c_gu_ptrs[b] = aligned_gate_up + row * (2LL * I_moe);

    a_dn_ptrs[b] = down_base + static_cast<long long>(e) * stride_dn;
    b_dn_ptrs[b] = aligned_act + row * I_moe;
    c_dn_ptrs[b] = aligned_out + row * H;
}

__global__ void reorder_moe_aligned_output_bf16_vec_kernel(
    const __nv_bfloat16* __restrict__ aligned_out,
    const std::int32_t* __restrict__ sorted_route_ids,
    __nv_bfloat16* __restrict__ route_out,
    int num_routes,
    int aligned_rows,
    int hidden_vec)
{
    const int hv = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (hv >= hidden_vec || row >= aligned_rows) return;
    const int route = sorted_route_ids[row];
    if (route >= num_routes) return;
    reinterpret_cast<uint4*>(route_out)[
        static_cast<long long>(route) * hidden_vec + hv] =
        reinterpret_cast<const uint4*>(aligned_out)[
            static_cast<long long>(row) * hidden_vec + hv];
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

void launch_moe_align_decode(
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
    // counts + offsets(+1) + fill, then 32 warp partials and one running base
    // for the block-wide scan.
    const std::size_t smem =
        static_cast<std::size_t>(3 * num_experts + 1 + 33) * sizeof(std::int32_t);
    moe_align_decode_kernel<<<1, BS, smem, stream>>>(
        topk_idx, sorted_route_ids, expert_ids, route_to_aligned_row,
        num_routes, num_experts, block_size, max_blocks);
}

void launch_moe_bucket_exact(
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

void launch_gather_moe_aligned_inputs_bf16(
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
    const auto* src = static_cast<const __nv_bfloat16*>(norm_x);
    auto* dst = static_cast<__nv_bfloat16*>(aligned_in);
    const bool vectorizable =
        (hidden % kMoeVecWidth) == 0 &&
        (reinterpret_cast<std::uintptr_t>(src) % 16) == 0 &&
        (reinterpret_cast<std::uintptr_t>(dst) % 16) == 0;
    if (vectorizable) {
        const int hidden_vec = hidden / kMoeVecWidth;
        const dim3 grid((hidden_vec + BS - 1) / BS, aligned_rows);
        gather_moe_aligned_inputs_bf16_vec_kernel<<<grid, BS, 0, stream>>>(
            src, sorted_route_ids, dst,
            num_routes, aligned_rows, top_k, hidden_vec);
        return;
    }
    const dim3 grid((hidden + BS - 1) / BS, aligned_rows);
    gather_moe_aligned_inputs_bf16_kernel<<<grid, BS, 0, stream>>>(
        src, sorted_route_ids, dst,
        num_routes, aligned_rows, top_k, hidden);
}

void launch_build_moe_ptrs_aligned_bf16(
    const std::int32_t* expert_ids,
    const void* gate_up_base,
    const void* down_base,
    const void* aligned_in,
    void* aligned_gate_up,
    void* aligned_act,
    void* aligned_out,
    const void** a_gu_ptrs,
    const void** b_gu_ptrs,
    void** c_gu_ptrs,
    const void** a_dn_ptrs,
    const void** b_dn_ptrs,
    void** c_dn_ptrs,
    int max_blocks,
    int block_size,
    int H,
    int I_moe,
    cudaStream_t stream)
{
    if (max_blocks <= 0) return;
    constexpr int BS = 256;
    const int grid = (max_blocks + BS - 1) / BS;
    build_moe_ptrs_aligned_bf16_kernel<<<grid, BS, 0, stream>>>(
        expert_ids,
        static_cast<const __nv_bfloat16*>(gate_up_base),
        static_cast<const __nv_bfloat16*>(down_base),
        static_cast<const __nv_bfloat16*>(aligned_in),
        static_cast<__nv_bfloat16*>(aligned_gate_up),
        static_cast<__nv_bfloat16*>(aligned_act),
        static_cast<__nv_bfloat16*>(aligned_out),
        reinterpret_cast<const __nv_bfloat16**>(a_gu_ptrs),
        reinterpret_cast<const __nv_bfloat16**>(b_gu_ptrs),
        reinterpret_cast<__nv_bfloat16**>(c_gu_ptrs),
        reinterpret_cast<const __nv_bfloat16**>(a_dn_ptrs),
        reinterpret_cast<const __nv_bfloat16**>(b_dn_ptrs),
        reinterpret_cast<__nv_bfloat16**>(c_dn_ptrs),
        max_blocks, block_size, H, I_moe);
}

void launch_reorder_moe_aligned_output_bf16(
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
    const auto* src = static_cast<const __nv_bfloat16*>(aligned_out);
    auto* dst = static_cast<__nv_bfloat16*>(route_out);
    const bool vectorizable =
        (hidden % kMoeVecWidth) == 0 &&
        (reinterpret_cast<std::uintptr_t>(src) % 16) == 0 &&
        (reinterpret_cast<std::uintptr_t>(dst) % 16) == 0;
    if (vectorizable) {
        const int hidden_vec = hidden / kMoeVecWidth;
        const dim3 grid((hidden_vec + BS - 1) / BS, aligned_rows);
        reorder_moe_aligned_output_bf16_vec_kernel<<<grid, BS, 0, stream>>>(
            src, sorted_route_ids, dst, num_routes, aligned_rows, hidden_vec);
        return;
    }
    const dim3 grid((hidden + BS - 1) / BS, aligned_rows);
    reorder_moe_aligned_output_bf16_kernel<<<grid, BS, 0, stream>>>(
        src, sorted_route_ids, dst, num_routes, aligned_rows, hidden);
}

}  // namespace pie_cuda_driver::kernels
