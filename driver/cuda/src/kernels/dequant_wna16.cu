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

// Unpacks eight INT4 weights out of one packed word into four fp32 pairs.
//
// The obvious way to turn a nibble into a float is `float(nibble - 8)`: mask,
// convert, subtract. That is three instructions for one weight, and these
// decode GEMVs are ALU-bound by a factor of four -- a loads-only version of
// `wna16_gate_up_decode_kernel` at Kimi K2.6's shapes runs at 5935 GB/s while
// the real kernel manages 1350 GB/s, so nothing matters here except the
// instruction count.
//
// So do it in the exponent instead. `0x4300` is bf16 128.0, whose low mantissa
// bits are zero; OR-ing a nibble into them yields bf16 `128 + q` exactly,
// because 128..143 all share that exponent. The mask, shift and OR fold into a
// single LOP3, and because the layout puts nibble j and nibble j+4 in the two
// halves of a 32-bit lane, one `__hsub2` against 136.0 (= 128 + 8) finishes
// two weights at once. Three instructions per pair, against ten.
__device__ __forceinline__ void wna16_unpack8_int4b8(unsigned word,
                                                     float2 out[4]) {
    constexpr unsigned kMagic = 0x43004300u;  // bf16 128.0, twice
    const __nv_bfloat162 kBias = __float2bfloat162_rn(136.0f);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        unsigned t = ((word >> (j * 4)) & 0x000f000fu) | kMagic;
        out[j] = __bfloat1622float2(
            __hsub2(*reinterpret_cast<__nv_bfloat162*>(&t), kBias));
    }
}

// One warp per output row for both halves of the fused gate/up projection.
//
// A block-per-row version spends eight `__syncthreads()` on the tree
// reduction for a handful of iterations of actual work; a warp reduces in
// five shuffles with no barrier at all.
//
// Two further instruction cuts, neither of which changes a single load:
// `wna16_unpack8_int4b8` for the dequant, and one 16-byte `float4` for the
// eight activations a word consumes -- converted once and shared by both
// halves, rather than eight scalar bf16 loads converted twice each. The group
// scale then multiplies the word's partial sum instead of every product, which
// is exact for the same reason it is cheaper: the scale is constant across the
// group. All three are algebraic rearrangements, so the result is bit-identical
// to the straightforward version; measured 87.0 -> 79.0 us for gate/up and
// 60.9 -> 49.2 us for down at Kimi's decode shapes, rel_l2 exactly 0.
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
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row = blockIdx.y * (blockDim.x >> 5) + warp_in_block;
    if (row >= intermediate) return;
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
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (int word_col = lane_id; word_col < words_per_row; word_col += 32) {
        const auto gate_word = static_cast<unsigned>(gate_packed[row_base + word_col]);
        const auto up_word = static_cast<unsigned>(up_packed[row_base + word_col]);
        const float4 xv = x4[word_col];
        const __nv_bfloat162* xh = reinterpret_cast<const __nv_bfloat162*>(&xv);
        float2 xp[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) xp[j] = __bfloat1622float2(xh[j]);
        // xp holds (k0+0,k0+1), (k0+2,k0+3), ...; the unpack pairs nibble j
        // with nibble j+4, so flatten to index by k directly.
        const float xe[8] = {xp[0].x, xp[0].y, xp[1].x, xp[1].y,
                             xp[2].x, xp[2].y, xp[3].x, xp[3].y};
        float2 gate_q[4], up_q[4];
        wna16_unpack8_int4b8(gate_word, gate_q);
        wna16_unpack8_int4b8(up_word, up_q);
        float gate_word_sum = 0.f;
        float up_word_sum = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            gate_word_sum = fmaf(gate_q[j].x, xe[j], gate_word_sum);
            gate_word_sum = fmaf(gate_q[j].y, xe[j + 4], gate_word_sum);
            up_word_sum = fmaf(up_q[j].x, xe[j], up_word_sum);
            up_word_sum = fmaf(up_q[j].y, xe[j + 4], up_word_sum);
        }
        const int group = (word_col * 8) / group_size;
        gate_acc = fmaf(gate_word_sum,
                        __bfloat162float(gate_scale[scale_base + group]),
                        gate_acc);
        up_acc = fmaf(up_word_sum,
                      __bfloat162float(up_scale[scale_base + group]), up_acc);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        gate_acc += __shfl_xor_sync(0xffffffffu, gate_acc, off);
        up_acc += __shfl_xor_sync(0xffffffffu, up_acc, off);
    }
    if (lane_id == 0) {
        const long long out_idx =
            static_cast<long long>(route) * intermediate + row;
        gate_out[out_idx] = __float2bfloat16(gate_acc);
        up_out[out_idx] = __float2bfloat16(up_acc);
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
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int h = blockIdx.x * (blockDim.x >> 5) + warp_in_block;
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
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (int word_col = lane_id; word_col < words_per_row; word_col += 32) {
        const auto word = static_cast<unsigned>(down_packed[row_base + word_col]);
        const float4 xv = x4[word_col];
        const __nv_bfloat162* xh = reinterpret_cast<const __nv_bfloat162*>(&xv);
        float2 xp[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) xp[j] = __bfloat1622float2(xh[j]);
        const float xe[8] = {xp[0].x, xp[0].y, xp[1].x, xp[1].y,
                             xp[2].x, xp[2].y, xp[3].x, xp[3].y};
        float2 q[4];
        wna16_unpack8_int4b8(word, q);
        float word_sum = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            word_sum = fmaf(q[j].x, xe[j], word_sum);
            word_sum = fmaf(q[j].y, xe[j + 4], word_sum);
        }
        const int group = (word_col * 8) / group_size;
        acc = fmaf(word_sum, __bfloat162float(down_scale[scale_base + group]),
                   acc);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, off);
    }
    if (lane_id == 0) {
        out[static_cast<long long>(route) * hidden + h] = __float2bfloat16(acc);
    }
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
    constexpr int GU_WARPS = DECODE_BLOCK / 32;
    const dim3 grid(routes, (intermediate + GU_WARPS - 1) / GU_WARPS);
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
    constexpr int WARPS = BS / 32;
    const dim3 grid((hidden + WARPS - 1) / WARPS, routes);
    wna16_down_decode_kernel<<<grid, BS, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(act_bf16),
        topk_idx,
        down_packed, down_scale,
        static_cast<__nv_bfloat16*>(out_bf16),
        top_k, hidden, intermediate, group_size);
}

}  // namespace pie_cuda_driver::kernels
