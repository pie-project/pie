#include "kernels/dequant_wna16.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>

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
    const int word_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int words_per_row = in_dim / 8;
    if (word_col >= words_per_row) return;

    // gridDim.y caps out at 65535, and an expert-stacked weight easily has more
    // rows than that, so the row axis is a grid-stride loop rather than a plain
    // block index.
    for (int row = blockIdx.y; row < out_dim; row += gridDim.y) {
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

// Unpacks eight INT4 weights out of one packed word into four fp16 pairs.
//
// The obvious way to turn a nibble into a float is `float(nibble - 8)`: mask,
// convert, subtract. That is three instructions for one weight, and these
// decode GEMVs are ALU-bound by a factor of four -- a loads-only version of
// `wna16_gate_up_decode_kernel` at Kimi K2.6's shapes runs at 5935 GB/s while
// the real kernel manages 1350 GB/s, so nothing matters here except the
// instruction count.
//
// So do it in the exponent instead. `0x6400` is fp16 1024.0, whose ten
// mantissa bits are zero; OR-ing a nibble into them yields fp16 `1024 + q`
// exactly, because 1024..1039 all share that exponent. The mask, shift and OR
// fold into a single LOP3, and because the layout puts nibble j and nibble j+4
// in the two halves of a 32-bit lane, one `__hsub2` against 1032.0 (= 1024 + 8)
// finishes two weights at once. Three instructions per pair, against ten.
//
// fp16 rather than bf16 for one reason: the products are then accumulated with
// `__hfma2`, two MACs per instruction and no format conversion at all, and
// fp16's 11 mantissa bits keep that accumulation honest. The bf16 version of
// exactly this kernel measures rel_l2 7.2e-3 against an fp32-accumulate
// reference -- about twice the bf16 *output's* own rounding, i.e. a real
// precision loss. The fp16 version measures 1.6e-3 for gate/up and is
// bit-exact for down, at the same speed.
__device__ __forceinline__ void wna16_unpack8_int4b8(unsigned word,
                                                     __half2 out[4]) {
    constexpr unsigned kMagic = 0x64006400u;  // fp16 1024.0, twice
    const __half2 kBias = __float2half2_rn(1032.0f);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        unsigned t = ((word >> (j * 4)) & 0x000f000fu) | kMagic;
        out[j] = __hsub2(*reinterpret_cast<__half2*>(&t), kBias);
    }
}

// Rearranges the eight activations a packed word consumes into the four
// (k, k+4) pairs the nibble trick wants, from a single 16-byte load.
//
// The eight fp16 values arrive as four 32-bit lanes {x1x0, x3x2, x5x4, x7x6};
// the pairs needed are (x0,x4), (x1,x5), (x2,x6), (x3,x7). Each is one PRMT
// picking two bytes from each of two source lanes. The alternative -- eight
// scalar loads and four `__halves2half2` -- is what made an earlier `__hfma2`
// attempt *lose* to the fp32 version despite doing half the arithmetic.
__device__ __forceinline__ void wna16_act_pairs(const float4& xv,
                                                __half2 xp[4]) {
    const unsigned* xu = reinterpret_cast<const unsigned*>(&xv);
    unsigned p0 = __byte_perm(xu[0], xu[2], 0x5410);
    unsigned p1 = __byte_perm(xu[0], xu[2], 0x7632);
    unsigned p2 = __byte_perm(xu[1], xu[3], 0x5410);
    unsigned p3 = __byte_perm(xu[1], xu[3], 0x7632);
    xp[0] = *reinterpret_cast<__half2*>(&p0);
    xp[1] = *reinterpret_cast<__half2*>(&p1);
    xp[2] = *reinterpret_cast<__half2*>(&p2);
    xp[3] = *reinterpret_cast<__half2*>(&p3);
}

// One warp per output row for both halves of the fused gate/up projection.
//
// A block-per-row version spends eight `__syncthreads()` on the tree
// reduction for a handful of iterations of actual work; a warp reduces in
// five shuffles with no barrier at all.
//
// The activation arrives as fp16 (staged once per MoE layer by
// `launch_bf16_to_fp16`, which costs one ~2 us launch against the ~15 us per
// layer this saves) so that the whole inner loop is `__hfma2`: two MACs per
// instruction, no format conversion anywhere. The group scale multiplies the
// word's partial sum instead of every product, which is exact for the same
// reason it is cheaper -- the scale is constant across the group -- and the
// promotion to fp32 happens once per word, so error cannot accumulate across
// the row. Measured against an fp32-accumulate reference at Kimi's decode
// shapes: gate/up 85.3 -> 69.0 us at rel_l2 1.6e-3, down 49.5 -> 38.3 us
// bit-exact.
__global__ void wna16_gate_up_decode_kernel(
    const __half* __restrict__ act,
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

    float gate_acc = 0.f;
    float up_acc = 0.f;
    const int words_per_row = hidden / 8;
    const int words_per_group = group_size / 8;
    const long long row_base = static_cast<long long>(row) * words_per_row;
    const long long scale_base =
        static_cast<long long>(row) * (hidden / group_size);
    const float4* x4 = reinterpret_cast<const float4*>(
        act + static_cast<long long>(token) * hidden);
    for (int word_col = lane_id; word_col < words_per_row; word_col += 32) {
        const auto gate_word = static_cast<unsigned>(gate_packed[row_base + word_col]);
        const auto up_word = static_cast<unsigned>(up_packed[row_base + word_col]);
        __half2 xp[4], gate_q[4], up_q[4];
        wna16_act_pairs(x4[word_col], xp);
        wna16_unpack8_int4b8(gate_word, gate_q);
        wna16_unpack8_int4b8(up_word, up_q);
        __half2 gate_word_sum = __float2half2_rn(0.f);
        __half2 up_word_sum = gate_word_sum;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            gate_word_sum = __hfma2(gate_q[j], xp[j], gate_word_sum);
            up_word_sum = __hfma2(up_q[j], xp[j], up_word_sum);
        }
        const int group = word_col / words_per_group;
        const float2 gf = __half22float2(gate_word_sum);
        const float2 uf = __half22float2(up_word_sum);
        gate_acc = fmaf(gf.x + gf.y,
                        __bfloat162float(gate_scale[scale_base + group]),
                        gate_acc);
        up_acc = fmaf(uf.x + uf.y,
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
    const __half* __restrict__ act,
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

    float acc = 0.f;
    const int words_per_row = intermediate / 8;
    const int words_per_group = group_size / 8;
    const long long row_base = static_cast<long long>(h) * words_per_row;
    const long long scale_base =
        static_cast<long long>(h) * (intermediate / group_size);
    const float4* x4 = reinterpret_cast<const float4*>(
        act + static_cast<long long>(route) * intermediate);
    for (int word_col = lane_id; word_col < words_per_row; word_col += 32) {
        const auto word = static_cast<unsigned>(down_packed[row_base + word_col]);
        __half2 xp[4], q[4];
        wna16_act_pairs(x4[word_col], xp);
        wna16_unpack8_int4b8(word, q);
        __half2 word_sum = __float2half2_rn(0.f);
#pragma unroll
        for (int j = 0; j < 4; ++j) word_sum = __hfma2(q[j], xp[j], word_sum);
        const float2 wf = __half22float2(word_sum);
        acc = fmaf(wf.x + wf.y,
                   __bfloat162float(
                       down_scale[scale_base + word_col / words_per_group]),
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
    constexpr int kMaxGridY = 65535;
    dim3 grid((words_per_row + BLOCK - 1) / BLOCK,
              static_cast<unsigned>(out_dim < kMaxGridY ? out_dim : kMaxGridY));
    dequant_wna16_int4b8_kernel<<<grid, BLOCK, 0, stream>>>(
        packed,
        static_cast<const __nv_bfloat16*>(scale_bf16),
        static_cast<__nv_bfloat16*>(out_bf16),
        out_dim,
        in_dim,
        group_size);
}

void launch_wna16_gate_up_decode_bf16(
    const void* act_fp16,
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
        static_cast<const __half*>(act_fp16),
        topk_idx,
        gate_packed, gate_scale,
        up_packed, up_scale,
        static_cast<__nv_bfloat16*>(gate_out_bf16),
        static_cast<__nv_bfloat16*>(up_out_bf16),
        top_k, hidden, intermediate, group_size);
}

void launch_wna16_down_decode_bf16(
    const void* act_fp16,
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
        static_cast<const __half*>(act_fp16),
        topk_idx,
        down_packed, down_scale,
        static_cast<__nv_bfloat16*>(out_bf16),
        top_k, hidden, intermediate, group_size);
}

// Stages a bf16 activation as fp16 for the W4A16 decode GEMVs.
//
// Those kernels want fp16 so their whole inner loop can be `__hfma2`. Doing
// the conversion inside them instead would cost four instructions per pair on
// every one of the ~2000 rows that read the same vector; doing it once here
// costs one pass over a few kilobytes plus a ~2 us launch, against ~15 us per
// layer saved. `float4`-wide because the conversion is pure bandwidth.
__global__ void bf16_to_fp16_kernel(const __nv_bfloat16* __restrict__ in,
                                    __half* __restrict__ out,
                                    long long n_vec8,
                                    long long n) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n_vec8; i += stride) {
        const float4 v = reinterpret_cast<const float4*>(in)[i];
        const __nv_bfloat162* src = reinterpret_cast<const __nv_bfloat162*>(&v);
        float4 o;
        __half2* dst = reinterpret_cast<__half2*>(&o);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(src[j]);
            dst[j] = __floats2half2_rn(f.x, f.y);
        }
        reinterpret_cast<float4*>(out)[i] = o;
    }
    for (long long i = n_vec8 * 8 + (long long)blockIdx.x * blockDim.x +
                       threadIdx.x;
         i < n; i += stride) {
        out[i] = __float2half(__bfloat162float(in[i]));
    }
}

void launch_bf16_to_fp16(const void* in_bf16, void* out_fp16,
                         std::size_t count, cudaStream_t stream) {
    if (count == 0) return;
    constexpr int BS = 256;
    const long long n = static_cast<long long>(count);
    const long long n_vec8 = n / 8;
    const long long units = n_vec8 > 0 ? n_vec8 : n;
    const int blocks = static_cast<int>(
        std::min<long long>((units + BS - 1) / BS, 1024));
    bf16_to_fp16_kernel<<<std::max(blocks, 1), BS, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(in_bf16),
        static_cast<__half*>(out_fp16), n_vec8, n);
}

}  // namespace pie_cuda_driver::kernels
