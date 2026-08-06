#include <cstdlib>
#include <type_traits>
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

// One warp per (route, slab of output rows). Each lane owns whole 32-element
// scale groups and loads them as a `uint4`: one group is exactly 16 packed
// bytes, so the widest possible load lines up with the smallest unit the
// scale is constant over. That constancy is also why the block scale can be
// folded in once per group in fp32 -- it keeps the fp16 accumulation depth
// at four and stops error walking down the row.
//
// The slab exists because the activation is re-read by every output row.
// One row per warp made the activation loads a third of the instruction
// stream and pinned `down` at 1.8 TB/s; four rows amortise each activation
// `float4` over four weight `uint4`s and take it to 3.3 TB/s. Going wider
// still helps at high route counts but starts losing at low ones, where
// there are no longer enough blocks to fill the machine.
//
// gate and up are adjacent rows of the *same* packed tensor (2i and 2i+1),
// so a slab of kPairs intermediate rows is 2*kPairs contiguous rows and
// needs no gather.
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
    constexpr int kPairs = 2;                      // intermediate rows per warp
    constexpr int kRows = 2 * kPairs;              // packed rows per warp
    const int route = blockIdx.x;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 =
        (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kPairs;
    if (row0 >= intermediate) return;
    const int token = route / top_k;
    const int expert = topk_idx[route];

    const std::uint8_t* packed = packed_ptrs[expert];
    const std::uint8_t* scales = scale_ptrs[expert];

    const int words_per_row = hidden / 8;          // 8 codes per 32-bit word
    const int groups_per_row = hidden / 32;
    // A slab can overhang the tail when `intermediate` is not a multiple of
    // kPairs. Clamp the overhanging rows onto the last real one: their
    // results are discarded at store time, and the alternative -- letting
    // the loads run past the tensor -- is an out-of-bounds read.
    int row_of[kRows];
#pragma unroll
    for (int p = 0; p < kPairs; ++p) {
        const int r = min(row0 + p, intermediate - 1);
        row_of[2 * p] = 2 * r;
        row_of[2 * p + 1] = 2 * r + 1;
    }

    const unsigned* w32 = reinterpret_cast<const unsigned*>(packed);
    const float4* x4 = reinterpret_cast<const float4*>(
        act + static_cast<long long>(token) * hidden);

    // Row r of the warp's slab is gate row (row0+r/2) for even r and its up
    // row for odd r -- the interleaving HF ships, so the whole slab is
    // 2*kPairs contiguous rows and no gather is needed.
    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.f;
    const uint4* wq = reinterpret_cast<const uint4*>(w32);
    for (int g = lane_id; g < groups_per_row; g += 32) {
        uint4 ww[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r)
            ww[r] = wq[static_cast<long long>(row_of[r]) *
                       (words_per_row >> 2) + g];
        __half2 sum[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) sum[r] = __float2half2_rn(0.f);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            __half2 xp[4];
            const float4 xv = x4[g * 4 + q];
            const unsigned* xu = reinterpret_cast<const unsigned*>(&xv);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                xp[j] = *reinterpret_cast<const __half2*>(&xu[j]);
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                __half2 qd[4];
                mxfp4_unpack8((&ww[r].x)[q], qd);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    sum[r] = __hfma2(qd[j], xp[j], sum[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const float2 f = __half22float2(sum[r]);
            acc[r] = fmaf(f.x + f.y,
                mxfp4_block_scale(scales[
                    static_cast<long long>(row_of[r]) * groups_per_row + g]),
                acc[r]);
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int r = 0; r < kRows; ++r)
            acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
    }
    if (lane_id == 0) {
        const auto* gb = gate_bias_ptrs != nullptr
            ? static_cast<const __nv_bfloat16*>(gate_bias_ptrs[expert])
            : nullptr;
        const auto* ub = gate_bias_ptrs != nullptr
            ? static_cast<const __nv_bfloat16*>(up_bias_ptrs[expert])
            : nullptr;
#pragma unroll
        for (int p = 0; p < kPairs; ++p) {
            const int row = row0 + p;
            if (row >= intermediate) break;
            float gv = acc[2 * p];
            float uv = acc[2 * p + 1];
            if (gb != nullptr) {
                gv += __bfloat162float(gb[row]);
                uv += __bfloat162float(ub[row]);
            }
            const long long o =
                static_cast<long long>(route) * intermediate + row;
            gate_out[o] = __float2bfloat16(gv);
            up_out[o] = __float2bfloat16(uv);
        }
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
    constexpr int kRows = 4;
    const int route = blockIdx.x;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 =
        (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kRows;
    if (row0 >= hidden) return;
    const int expert = topk_idx[route];

    const std::uint8_t* packed = packed_ptrs[expert];
    const std::uint8_t* scales = scale_ptrs[expert];

    const int words_per_row = intermediate / 8;
    const int groups_per_row = intermediate / 32;
    int row_of[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) row_of[r] = min(row0 + r, hidden - 1);

    const unsigned* w32 = reinterpret_cast<const unsigned*>(packed);
    const float4* x4 = reinterpret_cast<const float4*>(
        act + static_cast<long long>(route) * intermediate);

    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.f;
    const uint4* wq = reinterpret_cast<const uint4*>(w32);
    for (int g = lane_id; g < groups_per_row; g += 32) {
        uint4 ww[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r)
            ww[r] = wq[static_cast<long long>(row_of[r]) *
                       (words_per_row >> 2) + g];
        __half2 sum[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) sum[r] = __float2half2_rn(0.f);
#pragma unroll
        for (int qi = 0; qi < 4; ++qi) {
            __half2 xp[4];
            const float4 xv = x4[g * 4 + qi];
            const unsigned* xu = reinterpret_cast<const unsigned*>(&xv);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                xp[j] = *reinterpret_cast<const __half2*>(&xu[j]);
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                __half2 q[4];
                mxfp4_unpack8((&ww[r].x)[qi], q);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    sum[r] = __hfma2(q[j], xp[j], sum[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const float2 f = __half22float2(sum[r]);
            acc[r] = fmaf(f.x + f.y,
                mxfp4_block_scale(scales[
                    static_cast<long long>(row_of[r]) * groups_per_row + g]),
                acc[r]);
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int r = 0; r < kRows; ++r)
            acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
    }
    if (lane_id == 0) {
        const auto* bias = bias_ptrs != nullptr
            ? static_cast<const __nv_bfloat16*>(bias_ptrs[expert]) : nullptr;
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const int row = row0 + r;
            if (row >= hidden) break;
            float v = acc[r];
            if (bias != nullptr) v += __bfloat162float(bias[row]);
            out[static_cast<long long>(route) * hidden + row] =
                __float2bfloat16(v);
        }
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


// Expert-grouped variant of `mxfp4_moe_gate_up_decode_kernel`.
//
// The per-route kernel above gives one block to each (token, expert) route, so
// an expert chosen by T tokens has its weight slab streamed T times. Weight
// traffic therefore grows with the token count while the useful work does not,
// which is what makes decode throughput flat in batch size: at N=32/top_k=4
// there are 128 routes over at most 32 experts, so three quarters of the HBM
// traffic is re-reading slabs already read.
//
// Here a block owns (expert, row slab) and walks the expert's own route list,
// loading each weight group once and applying it to up to `kTok` tokens. The
// route list comes from `launch_moe_bucket_exact`, which already groups routes
// by expert; `counts` is its per-expert histogram and the block recovers its
// slice with an exclusive prefix over it (num_experts is 32-128, so this is
// cheaper than a second kernel and keeps the whole path device-side).
template <int kTok>
__global__ void mxfp4_moe_gate_up_decode_grouped_kernel(
    const __half* __restrict__ act,
    const std::int32_t* __restrict__ sorted_route_ids,
    const std::int32_t* __restrict__ counts,
    const std::uint8_t* const* __restrict__ packed_ptrs,
    const std::uint8_t* const* __restrict__ scale_ptrs,
    const void* const* __restrict__ gate_bias_ptrs,
    const void* const* __restrict__ up_bias_ptrs,
    __nv_bfloat16* __restrict__ gate_out,
    __nv_bfloat16* __restrict__ up_out,
    int top_k,
    int hidden,
    int intermediate,
    int num_experts)
{
    constexpr int kPairs = 2;
    constexpr int kRows = 2 * kPairs;
    const int expert = blockIdx.x;

    __shared__ int s_start;
    __shared__ int s_cnt;
    if (threadIdx.x == 0) {
        int st = 0;
        for (int e = 0; e < expert; ++e) st += counts[e];
        s_start = st;
        s_cnt = counts[expert];
    }
    __syncthreads();
    const int cnt = s_cnt;
    if (cnt == 0) return;

    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 = (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kPairs;
    if (row0 >= intermediate) return;

    const std::uint8_t* packed = packed_ptrs[expert];
    const std::uint8_t* scales = scale_ptrs[expert];
    const int words_per_row = hidden / 8;
    const int groups_per_row = hidden / 32;

    int row_of[kRows];
#pragma unroll
    for (int p = 0; p < kPairs; ++p) {
        const int r = min(row0 + p, intermediate - 1);
        row_of[2 * p] = 2 * r;
        row_of[2 * p + 1] = 2 * r + 1;
    }
    const uint4* wq = reinterpret_cast<const uint4*>(
        reinterpret_cast<const unsigned*>(packed));

    for (int base = 0; base < cnt; base += kTok) {
        const int nt = min(kTok, cnt - base);
        int route_of[kTok];
        const float4* x4[kTok];
#pragma unroll
        for (int t = 0; t < kTok; ++t) {
            const int idx = (t < nt) ? (s_start + base + t) : (s_start + base);
            route_of[t] = sorted_route_ids[idx];
            x4[t] = reinterpret_cast<const float4*>(
                act + static_cast<long long>(route_of[t] / top_k) * hidden);
        }

        float acc[kRows][kTok];
#pragma unroll
        for (int r = 0; r < kRows; ++r)
#pragma unroll
            for (int t = 0; t < kTok; ++t) acc[r][t] = 0.f;

        for (int g = lane_id; g < groups_per_row; g += 32) {
            uint4 ww[kRows];
#pragma unroll
            for (int r = 0; r < kRows; ++r)
                ww[r] = wq[static_cast<long long>(row_of[r]) *
                           (words_per_row >> 2) + g];
            __half2 sum[kRows][kTok];
#pragma unroll
            for (int r = 0; r < kRows; ++r)
#pragma unroll
                for (int t = 0; t < kTok; ++t) sum[r][t] = __float2half2_rn(0.f);

#pragma unroll
            for (int q = 0; q < 4; ++q) {
                // Unpack the weight quad once and reuse it for every token in
                // this pass -- the whole point of the grouping.
                __half2 qd[kRows][4];
#pragma unroll
                for (int r = 0; r < kRows; ++r)
                    mxfp4_unpack8((&ww[r].x)[q], qd[r]);
#pragma unroll
                for (int t = 0; t < kTok; ++t) {
                    if (t >= nt) break;
                    __half2 xp[4];
                    const float4 xv = x4[t][g * 4 + q];
                    const unsigned* xu = reinterpret_cast<const unsigned*>(&xv);
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        xp[j] = *reinterpret_cast<const __half2*>(&xu[j]);
#pragma unroll
                    for (int r = 0; r < kRows; ++r)
#pragma unroll
                        for (int j = 0; j < 4; ++j)
                            sum[r][t] = __hfma2(qd[r][j], xp[j], sum[r][t]);
                }
            }
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                const float sc = mxfp4_block_scale(scales[
                    static_cast<long long>(row_of[r]) * groups_per_row + g]);
#pragma unroll
                for (int t = 0; t < kTok; ++t) {
                    if (t >= nt) break;
                    const float2 f = __half22float2(sum[r][t]);
                    acc[r][t] = fmaf(f.x + f.y, sc, acc[r][t]);
                }
            }
        }

#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
            for (int r = 0; r < kRows; ++r)
#pragma unroll
                for (int t = 0; t < kTok; ++t)
                    acc[r][t] += __shfl_xor_sync(0xffffffffu, acc[r][t], off);
        }
        if (lane_id == 0) {
            const auto* gb = gate_bias_ptrs != nullptr
                ? static_cast<const __nv_bfloat16*>(gate_bias_ptrs[expert])
                : nullptr;
            const auto* ub = gate_bias_ptrs != nullptr
                ? static_cast<const __nv_bfloat16*>(up_bias_ptrs[expert])
                : nullptr;
            for (int t = 0; t < nt; ++t) {
#pragma unroll
                for (int p = 0; p < kPairs; ++p) {
                    const int row = row0 + p;
                    if (row >= intermediate) break;
                    float gv = acc[2 * p][t];
                    float uv = acc[2 * p + 1][t];
                    if (gb != nullptr) {
                        gv += __bfloat162float(gb[row]);
                        uv += __bfloat162float(ub[row]);
                    }
                    const long long o =
                        static_cast<long long>(route_of[t]) * intermediate + row;
                    gate_out[o] = __float2bfloat16(gv);
                    up_out[o] = __float2bfloat16(uv);
                }
            }
        }
    }
}

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
    const int pairs_per_block = warps * 2;   // kPairs = 2 rows per warp
    dim3 grid(num_tokens * top_k,
              (intermediate + pairs_per_block - 1) / pairs_per_block);
    mxfp4_moe_gate_up_decode_kernel<<<grid, kMxfp4DecodeBlock, 0, stream>>>(
        static_cast<const __half*>(act_fp16), topk_idx,
        gate_up_packed, gate_up_scales, gate_bias, up_bias,
        static_cast<__nv_bfloat16*>(gate_out_bf16),
        static_cast<__nv_bfloat16*>(up_out_bf16),
        top_k, hidden, intermediate);
}

void launch_mxfp4_moe_gate_up_decode_grouped_bf16(
    const void* act_fp16,
    const std::int32_t* sorted_route_ids,
    const std::int32_t* counts,
    const std::uint8_t* const* gate_up_packed,
    const std::uint8_t* const* gate_up_scales,
    const void* const* gate_bias,
    const void* const* up_bias,
    void* gate_out_bf16,
    void* up_out_bf16,
    int num_experts, int top_k, int hidden, int intermediate,
    cudaStream_t stream)
{
    if (num_experts <= 0 || top_k <= 0 || hidden <= 0 || intermediate <= 0) {
        return;
    }
    if (hidden % 32 != 0) return;
    const int warps = kMxfp4DecodeBlock / 32;
    const int pairs_per_block = warps * 2;
    // One block per (expert, row slab). Experts with no routes exit on their
    // first instruction, which is cheaper than compacting the list on device.
    dim3 grid(num_experts,
              (intermediate + pairs_per_block - 1) / pairs_per_block);
    // Tokens fused per weight pass. The weight bytes are read once per pass
    // and the FP4 unpack is amortised over the tokens in it, so this is the
    // knob that decides whether the kernel is unpack-bound or bandwidth-bound.
    // Swept with `driver/cuda/bench/moe_bench.cu`; PIE_MXFP4_MOE_KTOK overrides.
    static const int ktok = [] {
        const char* v = std::getenv("PIE_MXFP4_MOE_KTOK");
        return v != nullptr ? std::atoi(v) : 4;
    }();
    auto launch = [&](auto tag) {
        constexpr int kT = decltype(tag)::value;
        mxfp4_moe_gate_up_decode_grouped_kernel<kT>
            <<<grid, kMxfp4DecodeBlock, 0, stream>>>(
                static_cast<const __half*>(act_fp16), sorted_route_ids, counts,
                gate_up_packed, gate_up_scales, gate_bias, up_bias,
                static_cast<__nv_bfloat16*>(gate_out_bf16),
                static_cast<__nv_bfloat16*>(up_out_bf16),
                top_k, hidden, intermediate, num_experts);
    };
    switch (ktok) {
        case 1:  launch(std::integral_constant<int, 1>{});  return;
        case 2:  launch(std::integral_constant<int, 2>{});  return;
        case 8:  launch(std::integral_constant<int, 8>{});  return;
        case 16: launch(std::integral_constant<int, 16>{}); return;
        default: break;
    }
    mxfp4_moe_gate_up_decode_grouped_kernel<4>
        <<<grid, kMxfp4DecodeBlock, 0, stream>>>(
            static_cast<const __half*>(act_fp16), sorted_route_ids, counts,
            gate_up_packed, gate_up_scales, gate_bias, up_bias,
            static_cast<__nv_bfloat16*>(gate_out_bf16),
            static_cast<__nv_bfloat16*>(up_out_bf16),
            top_k, hidden, intermediate, num_experts);
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
    const int rows_per_warp = 4;
    const int rows_per_block = warps * rows_per_warp;
    dim3 grid(num_tokens * top_k,
              (hidden + rows_per_block - 1) / rows_per_block);
    mxfp4_moe_down_decode_kernel<<<grid, kMxfp4DecodeBlock, 0, stream>>>(
        static_cast<const __half*>(act_fp16), topk_idx,
        down_packed, down_scales, down_bias,
        static_cast<__nv_bfloat16*>(out_bf16),
        hidden, intermediate);
}

}  // namespace pie_cuda_driver::kernels
