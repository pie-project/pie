#include "kernels/moe_dispatch.hpp"

#include <cuda_bf16.h>

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

}  // namespace

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

}  // namespace pie_cuda_driver::kernels
