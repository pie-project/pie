#include "kernels/dequant_wna16.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

constexpr int DECODE_BLOCK = 256;

__global__ void dequant_wna16_int4b8_kernel(
    const std::int32_t* __restrict__ packed,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ out,
    int out_dim,
    int in_dim,
    int group_size)
{
    const int row = blockIdx.y;
    const int word_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int words_per_row = in_dim / 8;
    if (row >= out_dim || word_col >= words_per_row) return;

    const int word = packed[static_cast<long long>(row) * words_per_row + word_col];
    const int k_base = word_col * 8;
    __nv_bfloat16* row_out = out + static_cast<long long>(row) * in_dim;
    const __nv_bfloat16* row_scale =
        scale + static_cast<long long>(row) * (in_dim / group_size);

#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
        const int k = k_base + lane;
        const int nibble = (word >> (lane * 4)) & 0xF;
        const float q = static_cast<float>(nibble - 8);
        const float s = __bfloat162float(row_scale[k / group_size]);
        row_out[k] = __float2bfloat16(q * s);
    }
}

__device__ __forceinline__ float wna16_load_int4b8(
    const std::int32_t* __restrict__ packed,
    const __nv_bfloat16* __restrict__ scale,
    int row,
    int col,
    int in_dim,
    int group_size)
{
    const int words_per_row = in_dim / 8;
    const int word =
        packed[static_cast<long long>(row) * words_per_row + col / 8];
    const int nibble = (word >> ((col & 7) * 4)) & 0xF;
    const float q = static_cast<float>(nibble - 8);
    const float s = __bfloat162float(
        scale[static_cast<long long>(row) * (in_dim / group_size) +
              col / group_size]);
    return q * s;
}

__global__ void wna16_gate_up_decode_kernel(
    const __nv_bfloat16* __restrict__ act,
    const std::int32_t* __restrict__ topk_idx,
    const std::int32_t* const* __restrict__ gate_packed_ptrs,
    const void* const* __restrict__ gate_scale_ptrs,
    const std::int32_t* const* __restrict__ up_packed_ptrs,
    const void* const* __restrict__ up_scale_ptrs,
    __nv_bfloat16* __restrict__ gate_out,
    __nv_bfloat16* __restrict__ up_out,
    int top_k,
    int hidden,
    int intermediate,
    int group_size)
{
    const int route = blockIdx.x;
    const int row = blockIdx.y;
    const int token = route / top_k;
    const int expert = topk_idx[route];

    const auto* gate_packed = gate_packed_ptrs[expert];
    const auto* gate_scale =
        static_cast<const __nv_bfloat16*>(gate_scale_ptrs[expert]);
    const auto* up_packed = up_packed_ptrs[expert];
    const auto* up_scale =
        static_cast<const __nv_bfloat16*>(up_scale_ptrs[expert]);
    const auto* x = act + static_cast<long long>(token) * hidden;

    float gate_acc = 0.f;
    float up_acc = 0.f;
    const int words_per_row = hidden / 8;
    const int scales_per_row = hidden / group_size;
    const long long row_base = static_cast<long long>(row) * words_per_row;
    const long long scale_base = static_cast<long long>(row) * scales_per_row;
    for (int word_col = threadIdx.x; word_col < words_per_row;
         word_col += DECODE_BLOCK) {
        const int gate_word = gate_packed[row_base + word_col];
        const int up_word = up_packed[row_base + word_col];
        const float gate_s = __bfloat162float(
            gate_scale[scale_base + (word_col * 8) / group_size]);
        const float up_s = __bfloat162float(
            up_scale[scale_base + (word_col * 8) / group_size]);
#pragma unroll
        for (int lane = 0; lane < 8; ++lane) {
            const int k = word_col * 8 + lane;
            const float xv = __bfloat162float(x[k]);
            const int gate_nibble = (gate_word >> (lane * 4)) & 0xF;
            const int up_nibble = (up_word >> (lane * 4)) & 0xF;
            gate_acc += xv * (static_cast<float>(gate_nibble - 8) * gate_s);
            up_acc += xv * (static_cast<float>(up_nibble - 8) * up_s);
        }
    }

    __shared__ float gate_s[DECODE_BLOCK];
    __shared__ float up_s[DECODE_BLOCK];
    gate_s[threadIdx.x] = gate_acc;
    up_s[threadIdx.x] = up_acc;
    __syncthreads();
    for (int off = DECODE_BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            gate_s[threadIdx.x] += gate_s[threadIdx.x + off];
            up_s[threadIdx.x] += up_s[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const long long out_idx =
            static_cast<long long>(route) * intermediate + row;
        gate_out[out_idx] = __float2bfloat16(gate_s[0]);
        up_out[out_idx] = __float2bfloat16(up_s[0]);
    }
}

__global__ void wna16_down_decode_kernel(
    const __nv_bfloat16* __restrict__ act,
    const std::int32_t* __restrict__ topk_idx,
    const std::int32_t* const* __restrict__ down_packed_ptrs,
    const void* const* __restrict__ down_scale_ptrs,
    __nv_bfloat16* __restrict__ out,
    int top_k,
    int hidden,
    int intermediate,
    int group_size)
{
    const int route = blockIdx.y;
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= hidden) return;
    const int expert = topk_idx[route];
    const auto* down_packed = down_packed_ptrs[expert];
    const auto* down_scale =
        static_cast<const __nv_bfloat16*>(down_scale_ptrs[expert]);
    const auto* x = act + static_cast<long long>(route) * intermediate;

    float acc = 0.f;
    const int words_per_row = intermediate / 8;
    const int scales_per_row = intermediate / group_size;
    const long long row_base = static_cast<long long>(h) * words_per_row;
    const long long scale_base = static_cast<long long>(h) * scales_per_row;
    for (int word_col = 0; word_col < words_per_row; ++word_col) {
        const int word = down_packed[row_base + word_col];
        const float s = __bfloat162float(
            down_scale[scale_base + (word_col * 8) / group_size]);
#pragma unroll
        for (int lane = 0; lane < 8; ++lane) {
            const int i = word_col * 8 + lane;
            const int nibble = (word >> (lane * 4)) & 0xF;
            const float q = static_cast<float>(nibble - 8);
            acc += __bfloat162float(x[i]) * (q * s);
        }
    }
    out[static_cast<long long>(route) * hidden + h] = __float2bfloat16(acc);
}

}  // namespace

void launch_dequant_wna16_int4b8_to_bf16(
    const std::int32_t* packed,
    const void* scale_bf16,
    void* out_bf16,
    int out_dim,
    int in_dim,
    int group_size,
    cudaStream_t stream)
{
    if (out_dim <= 0 || in_dim <= 0 || group_size <= 0) return;
    if (in_dim % 8 != 0 || in_dim % group_size != 0) return;
    constexpr int BLOCK = 128;
    const int words_per_row = in_dim / 8;
    dim3 grid((words_per_row + BLOCK - 1) / BLOCK, out_dim);
    dequant_wna16_int4b8_kernel<<<grid, BLOCK, 0, stream>>>(
        packed,
        static_cast<const __nv_bfloat16*>(scale_bf16),
        static_cast<__nv_bfloat16*>(out_bf16),
        out_dim,
        in_dim,
        group_size);
}

void launch_wna16_gate_up_decode_bf16(
    const void* act_bf16,
    const std::int32_t* topk_idx,
    const std::int32_t* const* gate_packed,
    const void* const* gate_scale,
    const std::int32_t* const* up_packed,
    const void* const* up_scale,
    void* gate_out_bf16,
    void* up_out_bf16,
    int num_tokens,
    int top_k,
    int hidden,
    int intermediate,
    int group_size,
    cudaStream_t stream)
{
    const int routes = num_tokens * top_k;
    if (routes <= 0 || hidden <= 0 || intermediate <= 0) return;
    if (hidden % 8 != 0 || hidden % group_size != 0) return;
    const dim3 grid(routes, intermediate);
    wna16_gate_up_decode_kernel<<<grid, DECODE_BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(act_bf16),
        topk_idx,
        gate_packed, gate_scale,
        up_packed, up_scale,
        static_cast<__nv_bfloat16*>(gate_out_bf16),
        static_cast<__nv_bfloat16*>(up_out_bf16),
        top_k, hidden, intermediate, group_size);
}

void launch_wna16_down_decode_bf16(
    const void* act_bf16,
    const std::int32_t* topk_idx,
    const std::int32_t* const* down_packed,
    const void* const* down_scale,
    void* out_bf16,
    int num_tokens,
    int top_k,
    int hidden,
    int intermediate,
    int group_size,
    cudaStream_t stream)
{
    const int routes = num_tokens * top_k;
    if (routes <= 0 || hidden <= 0 || intermediate <= 0) return;
    if (intermediate % 8 != 0 || intermediate % group_size != 0) return;
    constexpr int BS = 256;
    const dim3 grid((hidden + BS - 1) / BS, routes);
    wna16_down_decode_kernel<<<grid, BS, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(act_bf16),
        topk_idx,
        down_packed, down_scale,
        static_cast<__nv_bfloat16*>(out_bf16),
        top_k, hidden, intermediate, group_size);
}

}  // namespace pie_cuda_driver::kernels
