//===-- dequant_fp4.cuh - the MXFP4 decoders, as device text -------------===//
//
// Four `__global__`s, the PRMT unpacker they share and the E2M1 lookup table
// behind it. `dequant_fp4.cu` is the four host launchers and the `kTok`
// dispatch that reads an environment variable; this file is what the GPU
// runs. There is exactly one definition of each, because two copies that
// agree today drift tomorrow and `norm/altup_aux` shipped a release proving
// it.
//
// # One row out of four kernels, and why the other three have none
//
// `dequant_mxfp4` launches `<<<out_dim, 128>>>` and strides by `blockDim.x`,
// which is `LaunchRule::RouteRows` -- one block per output row, block width
// picked from the row. The rule picks a different width than 128, and the
// kernel already took its stride from `blockDim.x` before this migration
// touched it, so nothing about the loop changes.
//
// The three MoE decode GEMVs have no rule and cannot get one:
//
//   * they launch a 2-D grid whose second axis is a SLAB of output rows --
//     `(routes, ceil(intermediate / pairs_per_block))` -- which no rule
//     states, and
//   * they are templated on `int`, not on a type. `DeviceKernel::instantiation`
//     spells `path<elem>` with `elem` a type path, so a kernel whose template
//     parameter is `kPairs = 4` cannot be named by a row at all.
//
// The second reason is the binding one: even with a rule, a row could not
// address them. They are here because the recipe is that a `.cuh` holds the
// family's device text -- a kernel left behind in the `.cu` is the second
// copy this split exists to prevent -- and NVRTC compiles them, which is
// itself worth having measured.
//
// # The fp16 packed arithmetic, and why the include is guarded
//
// `__hfma2`, `__half22float2` and `__float2half2_rn` are hardware
// instructions, not arithmetic `pie_device.cuh` could restate:
// `new-horizon.md` §10.5 lists them as the reason this file was BLOCKED, and
// §8 refused to rewrite the bodies for a migration. §15 closed it from the
// other side -- `pie_half2.cuh` is the honest name for the packed-half shim,
// measured bit-identical to nvcc over 32,945,058 comparisons including every
// architecture fallback -- so the intrinsics stay spelled the way they were
// and only the include is chosen per compiler.
//
// The guard is not a stopgap. The archive crate's
// `kernels-cuda/csrc/CMakeLists.txt` put this directory on the ahead-of-time
// build's path with `-iquote`, which answers
// `#include "..."` and is never searched for `#include <...>` -- precisely
// so that a shim wearing NVIDIA's filename cannot shadow the real
// `cuda_fp16.h` for every translation unit in the tree. NVRTC has no include
// path at all and resolves `pie_half2.cuh` out of the carried set.
//
// `<cstdlib>` and `<type_traits>` are gone from this file and stayed in the
// `.cu`: they were there for `std::getenv`, `std::atoi` and
// `std::integral_constant`, which are the `kTok` dispatch's and entirely
// host-side. NVRTC ships no C++ standard library -- 0 of 31 standard headers
// answered when it was measured -- so a device header that named one compiles
// nowhere.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

#ifdef __CUDACC_RTC__
#include "prelude/half2.cuh"
#else
#include <cuda_fp16.h>
#endif

namespace pie::quant {


// E2M1 codepoint → fp32 LUT. Index is the 4-bit code (high bit = sign,
// next two = exponent biased at 1, low bit = mantissa). Matches OCP's
// MX FP4 spec.
__device__ __constant__ float kFp4Lut[16] = {
     0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
    -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f,
};

// One block per output row; each thread strides by 32 elements (the
// block-scale granularity). Per-32-element block: read the E8M0 scale,
// dequantize 16 packed-byte pairs into 32 elements of `T`.
//
// A template where the original was bf16-only, for `elementwise.cuh`'s
// reason: the ahead-of-time build had to choose its instantiations and
// nothing here is bf16-specific past the narrowing store, so the fp16
// dequantiser costs a row rather than a translation unit.
template <class T>
__global__ void dequant_mxfp4(
    const u8* __restrict__ packed,
    const u8* __restrict__ block_scale,
    T*      __restrict__ out,
    int                 in_dim)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int blocks_per_row = in_dim / 32;

    const u8* row_packed = packed + static_cast<long long>(row) * (in_dim / 2);
    const u8* row_scale  = block_scale + static_cast<long long>(row) * blocks_per_row;
    T*      row_out    = out + static_cast<long long>(row) * in_dim;

    for (int blk = tid; blk < blocks_per_row; blk += blockDim.x) {
        const u8 e8m0 = row_scale[blk];
        // E8M0: byte b → scale = 2^(b - 127). 0xFF is reserved for NaN
        // in the MX spec; we let exp2f overflow → +inf and downstream
        // bf16 saturation handle it (matches reference impls).
        const float scale = exp2f(static_cast<float>(static_cast<int>(e8m0)) - 127.f);

        const int packed_base = blk * 16;       // 16 bytes hold 32 fp4 codes
        const int out_base    = blk * 32;
        for (int i = 0; i < 16; ++i) {
            const u8 b = row_packed[packed_base + i];
            const float v_lo = kFp4Lut[b & 0xF] * scale;
            const float v_hi = kFp4Lut[b >> 4]  * scale;
            row_out[out_base + 2 * i + 0] = Elem<T>::from_f32(v_lo);
            row_out[out_base + 2 * i + 1] = Elem<T>::from_f32(v_hi);
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
__device__ __forceinline__ float mxfp4_block_scale(u8 b) {
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
// `act_out_fp16`, when non-null, receives the gpt-oss GLU of the two halves --
// clamp, quickgelu on the gate, `(up + 1) *` -- in fp16, which is what the down
// projection reads. Both terms are already in registers here, so the activation
// costs a few instructions on values that would otherwise be written to HBM,
// read back by a glu kernel, written again, and read a third time by a cast.
// vLLM does the same thing from the other side: its MoE matmul is
// `_matmul_ogs_..._swiglu`, one kernel with the activation in the epilogue.
template <int kPairsT>
__global__ void mxfp4_moe_gate_up_decode(
    const __half* __restrict__ act,
    const i32* __restrict__ topk_idx,
    const u8* const* __restrict__ packed_ptrs,
    const u8* const* __restrict__ scale_ptrs,
    const void* const* __restrict__ gate_bias_ptrs,
    const void* const* __restrict__ up_bias_ptrs,
    bf16* __restrict__ gate_out,
    bf16* __restrict__ up_out,
    __half* __restrict__ act_out_fp16,
    float glu_limit,
    float glu_alpha,
    int top_k,
    int hidden,
    int intermediate)
{
    // Intermediate rows per warp. Every extra pair reuses the activation
    // vector one more time and gives the unpack more independent work to hide
    // behind, which is what this kernel is short of: at 2 it sustains about
    // 1.4 TB/s against an HBM roofline near 3.
    constexpr int kPairs = kPairsT;
    constexpr int kRows = 2 * kPairs;              // packed rows per warp
    const int route = blockIdx.x;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 =
        (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kPairs;
    if (row0 >= intermediate) return;
    const int token = route / top_k;
    const int expert = topk_idx[route];

    const u8* packed = packed_ptrs[expert];
    const u8* scales = scale_ptrs[expert];

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
            ? static_cast<const bf16*>(gate_bias_ptrs[expert])
            : nullptr;
        const auto* ub = gate_bias_ptrs != nullptr
            ? static_cast<const bf16*>(up_bias_ptrs[expert])
            : nullptr;
#pragma unroll
        for (int p = 0; p < kPairs; ++p) {
            const int row = row0 + p;
            if (row >= intermediate) break;
            float gv = acc[2 * p];
            float uv = acc[2 * p + 1];
            if (gb != nullptr) {
                gv += bf16_to_f32(gb[row]);
                uv += bf16_to_f32(ub[row]);
            }
            const long long o =
                static_cast<long long>(route) * intermediate + row;
            if (act_out_fp16 != nullptr) {
                // Same arithmetic and the same order as
                // `gpt_oss_glu_bf16_kernel`, which read these two back from
                // HBM to do it.
                const float g = fminf(gv, glu_limit);
                const float u = fminf(fmaxf(uv, -glu_limit), glu_limit);
                const float glu = g / (1.f + __expf(-glu_alpha * g));
                act_out_fp16[o] = __float2half((u + 1.f) * glu);
            } else {
                gate_out[o] = f32_to_bf16(gv);
                up_out[o] = f32_to_bf16(uv);
            }
        }
    }
}

template <int kRowsT>
__global__ void mxfp4_moe_down_decode(
    const __half* __restrict__ act,
    const i32* __restrict__ topk_idx,
    const u8* const* __restrict__ packed_ptrs,
    const u8* const* __restrict__ scale_ptrs,
    const void* const* __restrict__ bias_ptrs,
    bf16* __restrict__ out,
    int hidden,
    int intermediate)
{
    constexpr int kRows = kRowsT;
    const int route = blockIdx.x;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 =
        (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kRows;
    if (row0 >= hidden) return;
    const int expert = topk_idx[route];

    const u8* packed = packed_ptrs[expert];
    const u8* scales = scale_ptrs[expert];

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
            ? static_cast<const bf16*>(bias_ptrs[expert]) : nullptr;
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const int row = row0 + r;
            if (row >= hidden) break;
            float v = acc[r];
            if (bias != nullptr) v += bf16_to_f32(bias[row]);
            out[static_cast<long long>(route) * hidden + row] =
                f32_to_bf16(v);
        }
    }
}

// Expert-grouped variant of `mxfp4_moe_gate_up_decode`.
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
// route list comes from `kernels::moe::moe_bucket_exact`, which already groups routes
// by expert; `counts` is its per-expert histogram and the block recovers its
// slice with an exclusive prefix over it (num_experts is 32-128, so this is
// cheaper than a second kernel and keeps the whole path device-side).
template <int kTok>
__global__ void mxfp4_moe_gate_up_decode_grouped(
    const __half* __restrict__ act,
    const i32* __restrict__ sorted_route_ids,
    const i32* __restrict__ counts,
    const u8* const* __restrict__ packed_ptrs,
    const u8* const* __restrict__ scale_ptrs,
    const void* const* __restrict__ gate_bias_ptrs,
    const void* const* __restrict__ up_bias_ptrs,
    bf16* __restrict__ gate_out,
    bf16* __restrict__ up_out,
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

    const u8* packed = packed_ptrs[expert];
    const u8* scales = scale_ptrs[expert];
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
                ? static_cast<const bf16*>(gate_bias_ptrs[expert])
                : nullptr;
            const auto* ub = gate_bias_ptrs != nullptr
                ? static_cast<const bf16*>(up_bias_ptrs[expert])
                : nullptr;
            for (int t = 0; t < nt; ++t) {
#pragma unroll
                for (int p = 0; p < kPairs; ++p) {
                    const int row = row0 + p;
                    if (row >= intermediate) break;
                    float gv = acc[2 * p][t];
                    float uv = acc[2 * p + 1][t];
                    if (gb != nullptr) {
                        gv += bf16_to_f32(gb[row]);
                        uv += bf16_to_f32(ub[row]);
                    }
                    const long long o =
                        static_cast<long long>(route_of[t]) * intermediate + row;
                    gate_out[o] = f32_to_bf16(gv);
                    up_out[o] = f32_to_bf16(uv);
                }
            }
        }
    }
}

}  // namespace pie::quant
