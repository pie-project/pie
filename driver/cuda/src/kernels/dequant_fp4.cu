#include "kernels/dequant_fp4.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace pie_cuda_driver::kernels {

namespace {

// E2M1 codepoint → fp32 LUT. Index is the 4-bit code (high bit = sign,
// next two = exponent biased at 1, low bit = mantissa). Matches OCP's
// MX FP4 spec.
__device__ __constant__ float kFp4Lut[16] = {
     0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
    -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f,
};

// One block per output row; each thread strides by 32 elements (the
// block-scale granularity). Per-32-element block: read the E8M0 scale,
// dequantize 16 packed-byte pairs into 32 bf16 elements.
__global__ void dequant_mxfp4_kernel(
    const std::uint8_t* __restrict__ packed,
    const std::uint8_t* __restrict__ block_scale,
    __nv_bfloat16*      __restrict__ out,
    int                 in_dim)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int blocks_per_row = in_dim / 32;

    const std::uint8_t* row_packed = packed + static_cast<long long>(row) * (in_dim / 2);
    const std::uint8_t* row_scale  = block_scale + static_cast<long long>(row) * blocks_per_row;
    __nv_bfloat16*      row_out    = out + static_cast<long long>(row) * in_dim;

    for (int blk = tid; blk < blocks_per_row; blk += blockDim.x) {
        const std::uint8_t e8m0 = row_scale[blk];
        // E8M0: byte b → scale = 2^(b - 127). 0xFF is reserved for NaN
        // in the MX spec; we let exp2f overflow → +inf and downstream
        // bf16 saturation handle it (matches reference impls).
        const float scale = exp2f(static_cast<float>(static_cast<int>(e8m0)) - 127.f);

        const int packed_base = blk * 16;       // 16 bytes hold 32 fp4 codes
        const int out_base    = blk * 32;
        for (int i = 0; i < 16; ++i) {
            const std::uint8_t b = row_packed[packed_base + i];
            const float v_lo = kFp4Lut[b & 0xF] * scale;
            const float v_hi = kFp4Lut[b >> 4]  * scale;
            row_out[out_base + 2 * i + 0] = __float2bfloat16(v_lo);
            row_out[out_base + 2 * i + 1] = __float2bfloat16(v_hi);
        }
    }
}


// Decodes the eight E2M1 codes packed into one 32-bit word into four fp16
// pairs, with no lookup table in memory and no branches.
//
// The eight fp16 values MXFP4 can hold all have a zero low byte:
// {0x0000, 0x3800, 0x3C00, 0x3E00, 0x4000, 0x4200, 0x4400, 0x4600} for the
// positives, and negating one only sets bit 15, i.e. ORs 0x80 into the same
// high byte. So the whole 16-entry table is eight *bytes*, which is exactly
// what one PRMT indexes: `__byte_perm(T0, T1, sel)` treats {T0,T1} as an
// 8-byte array and selects four of them by the four nibbles of `sel`.
//
// A packed word holds eight codes at nibble positions 0..7, so its low and
// high halves are already valid four-nibble selectors — no shuffling is
// needed to build them. Masking with 0x7777 leaves the magnitude index and
// a second PRMT against {0x00000000, 0x80808080} turns the sign bits into
// 0x00/0x80 bytes to OR in. Two more PRMTs per half spread the four high
// bytes into four 16-bit lanes.
//
// Sixteen instructions for eight weights. The arithmetic alternative --
// `bits = ((c & 7) << 9) + 0x3800` -- is exact for six of the eight
// magnitudes but wrong for both subnormal codes (0 and 0.5), and patching
// those costs more than the PRMTs do.
//
// The pairing that falls out is (element 2j, element 2j+1), which is also
// how the fp16 activation sits in registers after a single float4 load. The
// INT4 sibling in `dequant_wna16.cu` has to shuffle its activation into
// (j, j+4) pairs; here that work does not exist.
__device__ __forceinline__ void mxfp4_unpack8(unsigned word, __half2 out[4]) {
    constexpr unsigned kMagHi01234567 = 0x3E3C3800u;  // codes 0..3 high bytes
    constexpr unsigned kMagHi4567     = 0x46444240u;  // codes 4..7 high bytes
    constexpr unsigned kSignBytes     = 0x80808080u;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const unsigned sel = (word >> (half * 16)) & 0xFFFFu;
        const unsigned mag =
            __byte_perm(kMagHi01234567, kMagHi4567, sel & 0x7777u);
        const unsigned sgn =
            __byte_perm(0u, kSignBytes, (sel & 0x8888u) >> 1);
        const unsigned hi = mag | sgn;
        const unsigned a = __byte_perm(hi, 0u, 0x1404u);
        const unsigned b = __byte_perm(hi, 0u, 0x3424u);
        out[half * 2 + 0] = *reinterpret_cast<const __half2*>(&a);
        out[half * 2 + 1] = *reinterpret_cast<const __half2*>(&b);
    }
}

// E8M0 byte b denotes 2^(b-127), which is a float's exponent field verbatim.
// b == 0 is 2^-127; building it from the bit pattern would give +0 instead,
// so fall back to exp2f there rather than silently zeroing a block.
__device__ __forceinline__ float mxfp4_block_scale(std::uint8_t b) {
    return b == 0 ? exp2f(-127.f)
                  : __int_as_float(static_cast<int>(b) << 23);
}

// One warp per (route, intermediate row). Each warp walks the row's packed
// words with a stride of 32, accumulating in fp16 pairs and folding the
// block scale in once per 32-element group -- exact, because the scale is
// constant across the group, and it keeps the promotion to fp32 at one per
// group so error cannot walk down the row.
//
// gate and up are two rows of the *same* packed tensor (2i and 2i+1), so
// one warp covers both and the expert's base pointers are loaded once.
__global__ void mxfp4_moe_gate_up_decode_kernel(
    const __half* __restrict__ act,
    const std::int32_t* __restrict__ topk_idx,
    const std::uint8_t* const* __restrict__ packed_ptrs,
    const std::uint8_t* const* __restrict__ scale_ptrs,
    const void* const* __restrict__ gate_bias_ptrs,
    const void* const* __restrict__ up_bias_ptrs,
    __nv_bfloat16* __restrict__ gate_out,
    __nv_bfloat16* __restrict__ up_out,
    int top_k,
    int hidden,
    int intermediate)
{
    const int route = blockIdx.x;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row = blockIdx.y * (blockDim.x >> 5) + warp_in_block;
    if (row >= intermediate) return;
    const int token = route / top_k;
    const int expert = topk_idx[route];

    const std::uint8_t* packed = packed_ptrs[expert];
    const std::uint8_t* scales = scale_ptrs[expert];

    const int words_per_row = hidden / 8;          // 8 codes per 32-bit word
    const int groups_per_row = hidden / 32;
    constexpr int kWordsPerGroup = 4;              // 32 codes / 8 per word
    const long long gate_base =
        static_cast<long long>(2 * row) * words_per_row;
    const long long up_base = gate_base + words_per_row;
    const long long gate_scale_base =
        static_cast<long long>(2 * row) * groups_per_row;
    const long long up_scale_base = gate_scale_base + groups_per_row;

    const unsigned* w32 = reinterpret_cast<const unsigned*>(packed);
    const float4* x4 = reinterpret_cast<const float4*>(
        act + static_cast<long long>(token) * hidden);

    float gate_acc = 0.f;
    float up_acc = 0.f;
    for (int word_col = lane_id; word_col < words_per_row; word_col += 32) {
        __half2 xp[4], gq[4], uq[4];
        const float4 xv = x4[word_col];
        const unsigned* xu = reinterpret_cast<const unsigned*>(&xv);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            xp[j] = *reinterpret_cast<const __half2*>(&xu[j]);
        }
        mxfp4_unpack8(w32[gate_base + word_col], gq);
        mxfp4_unpack8(w32[up_base + word_col], uq);
        __half2 gsum = __float2half2_rn(0.f);
        __half2 usum = gsum;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            gsum = __hfma2(gq[j], xp[j], gsum);
            usum = __hfma2(uq[j], xp[j], usum);
        }
        const int group = word_col / kWordsPerGroup;
        const float2 gf = __half22float2(gsum);
        const float2 uf = __half22float2(usum);
        gate_acc = fmaf(gf.x + gf.y,
                        mxfp4_block_scale(scales[gate_scale_base + group]),
                        gate_acc);
        up_acc = fmaf(uf.x + uf.y,
                      mxfp4_block_scale(scales[up_scale_base + group]),
                      up_acc);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        gate_acc += __shfl_xor_sync(0xffffffffu, gate_acc, off);
        up_acc += __shfl_xor_sync(0xffffffffu, up_acc, off);
    }
    if (lane_id == 0) {
        if (gate_bias_ptrs != nullptr) {
            gate_acc += __bfloat162float(
                static_cast<const __nv_bfloat16*>(gate_bias_ptrs[expert])[row]);
            up_acc += __bfloat162float(
                static_cast<const __nv_bfloat16*>(up_bias_ptrs[expert])[row]);
        }
        const long long out_idx =
            static_cast<long long>(route) * intermediate + row;
        gate_out[out_idx] = __float2bfloat16(gate_acc);
        up_out[out_idx] = __float2bfloat16(up_acc);
    }
}

__global__ void mxfp4_moe_down_decode_kernel(
    const __half* __restrict__ act,
    const std::int32_t* __restrict__ topk_idx,
    const std::uint8_t* const* __restrict__ packed_ptrs,
    const std::uint8_t* const* __restrict__ scale_ptrs,
    const void* const* __restrict__ bias_ptrs,
    __nv_bfloat16* __restrict__ out,
    int hidden,
    int intermediate)
{
    const int route = blockIdx.x;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row = blockIdx.y * (blockDim.x >> 5) + warp_in_block;
    if (row >= hidden) return;
    const int expert = topk_idx[route];

    const std::uint8_t* packed = packed_ptrs[expert];
    const std::uint8_t* scales = scale_ptrs[expert];

    const int words_per_row = intermediate / 8;
    const int groups_per_row = intermediate / 32;
    constexpr int kWordsPerGroup = 4;
    const long long row_base = static_cast<long long>(row) * words_per_row;
    const long long scale_base =
        static_cast<long long>(row) * groups_per_row;

    const unsigned* w32 = reinterpret_cast<const unsigned*>(packed);
    const float4* x4 = reinterpret_cast<const float4*>(
        act + static_cast<long long>(route) * intermediate);

    float acc = 0.f;
    for (int word_col = lane_id; word_col < words_per_row; word_col += 32) {
        __half2 xp[4], q[4];
        const float4 xv = x4[word_col];
        const unsigned* xu = reinterpret_cast<const unsigned*>(&xv);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            xp[j] = *reinterpret_cast<const __half2*>(&xu[j]);
        }
        mxfp4_unpack8(w32[row_base + word_col], q);
        __half2 sum = __float2half2_rn(0.f);
#pragma unroll
        for (int j = 0; j < 4; ++j) sum = __hfma2(q[j], xp[j], sum);
        const int group = word_col / kWordsPerGroup;
        const float2 f = __half22float2(sum);
        acc = fmaf(f.x + f.y,
                   mxfp4_block_scale(scales[scale_base + group]), acc);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, off);
    }
    if (lane_id == 0) {
        if (bias_ptrs != nullptr) {
            const auto* bias =
                static_cast<const __nv_bfloat16*>(bias_ptrs[expert]);
            acc += __bfloat162float(bias[row]);
        }
        out[static_cast<long long>(route) * hidden + row] =
            __float2bfloat16(acc);
    }
}

}  // namespace

void launch_dequant_mxfp4_to_bf16(
    const std::uint8_t* packed, const std::uint8_t* block_scale,
    void* out, int out_dim, int in_dim, cudaStream_t stream)
{
    if (out_dim <= 0 || in_dim <= 0) return;
    if (in_dim % 32 != 0) return;
    constexpr int BLOCK = 128;
    dim3 grid(out_dim);
    dim3 block(BLOCK);
    dequant_mxfp4_kernel<<<grid, block, 0, stream>>>(
        packed, block_scale,
        static_cast<__nv_bfloat16*>(out), in_dim);
}


namespace {
constexpr int kMxfp4DecodeBlock = 128;  // four warps, one output row each
}  // namespace

void launch_mxfp4_moe_gate_up_decode_bf16(
    const void* act_fp16,
    const std::int32_t* topk_idx,
    const std::uint8_t* const* gate_up_packed,
    const std::uint8_t* const* gate_up_scales,
    const void* const* gate_bias,
    const void* const* up_bias,
    void* gate_out_bf16,
    void* up_out_bf16,
    int num_tokens, int top_k, int hidden, int intermediate,
    cudaStream_t stream)
{
    if (num_tokens <= 0 || top_k <= 0 || hidden <= 0 || intermediate <= 0) {
        return;
    }
    if (hidden % 32 != 0) return;
    const int warps = kMxfp4DecodeBlock / 32;
    dim3 grid(num_tokens * top_k, (intermediate + warps - 1) / warps);
    mxfp4_moe_gate_up_decode_kernel<<<grid, kMxfp4DecodeBlock, 0, stream>>>(
        static_cast<const __half*>(act_fp16), topk_idx,
        gate_up_packed, gate_up_scales, gate_bias, up_bias,
        static_cast<__nv_bfloat16*>(gate_out_bf16),
        static_cast<__nv_bfloat16*>(up_out_bf16),
        top_k, hidden, intermediate);
}

void launch_mxfp4_moe_down_decode_bf16(
    const void* act_fp16,
    const std::int32_t* topk_idx,
    const std::uint8_t* const* down_packed,
    const std::uint8_t* const* down_scales,
    const void* const* down_bias,
    void* out_bf16,
    int num_tokens, int top_k, int hidden, int intermediate,
    cudaStream_t stream)
{
    if (num_tokens <= 0 || top_k <= 0 || hidden <= 0 || intermediate <= 0) {
        return;
    }
    if (intermediate % 32 != 0) return;
    const int warps = kMxfp4DecodeBlock / 32;
    dim3 grid(num_tokens * top_k, (hidden + warps - 1) / warps);
    mxfp4_moe_down_decode_kernel<<<grid, kMxfp4DecodeBlock, 0, stream>>>(
        static_cast<const __half*>(act_fp16), topk_idx,
        down_packed, down_scales, down_bias,
        static_cast<__nv_bfloat16*>(out_bf16),
        hidden, intermediate);
}

}  // namespace pie_cuda_driver::kernels
