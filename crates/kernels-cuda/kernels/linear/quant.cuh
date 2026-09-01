#pragma once

#include "prelude/device.cuh"

#ifdef __CUDACC_RTC__
#include "prelude/half2.cuh"
#include "prelude/fp8.cuh"
#else
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#endif

namespace pie::linear {

constexpr int kBlock = 256;

struct fp8_e4m3 {
    using store = u8;

    static __host__ __device__ __forceinline__ float max_abs() { return 448.f; }
    static __device__ __forceinline__ store narrow(float v) {
        return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    }
};

struct int8_sym {
    using store = i8;
    static __host__ __device__ __forceinline__ float max_abs() { return 127.f; }
    static __device__ __forceinline__ store narrow(float v) {
        int q = static_cast<int>(rintf(v));
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        return static_cast<store>(q);
    }
};

__device__ __forceinline__ float row_absmax(float local, float* smem, int tid) {
    for (int off = 16; off > 0; off >>= 1) {
        const float other = __shfl_down_sync(0xffffffff, local, off);
        if (other > local) local = other;
    }
    const int lane = tid & 31;
    const int warp = tid / 32;
    if (lane == 0) smem[warp] = local;
    __syncthreads();
    if (warp == 0) {
        local = (tid < kBlock / 32) ? smem[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1) {
            const float other = __shfl_down_sync(0xffffffff, local, off);
            if (other > local) local = other;
        }
        if (lane == 0) smem[0] = local;
    }
    __syncthreads();
    return smem[0];
}

template <class Fmt>
__global__ void quant_flat(
    const bf16* __restrict__ W,
    typename Fmt::store* __restrict__ out,
    float scale_inv,
    usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = Fmt::narrow(bf16_to_f32(W[i]) * scale_inv);
}

template <class Fmt>
__global__ void absmax_to_scale_inv(float* x, i32 n) {
    const i32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    x[i] = (v > 0.f) ? (v / Fmt::max_abs()) : 1.f;
}

template <class T>
__global__ void absmax_per_row(
    const T* __restrict__ W, float* __restrict__ absmax_out, i32 cols) {
    const int tid = threadIdx.x;
    extern __shared__ float warp_max[];

    const usize row_off = static_cast<usize>(blockIdx.x) * cols;
    float local = 0.f;
    for (i32 j = tid; j < cols; j += kBlock) {
        const float v = fabsf(Elem<T>::to_f32(W[row_off + j]));
        if (v > local) local = v;
    }
    const float row_max = row_absmax(local, warp_max, tid);
    if (tid == 0) absmax_out[blockIdx.x] = row_max;
}

__global__ void absmax_bf16(
    const bf16* __restrict__ W, float* __restrict__ out, usize n) {
    __shared__ float warp_max[kBlock / 32];
    const unsigned tid = threadIdx.x;
    const unsigned warp = tid / 32;
    const unsigned lane = tid & 31;
    usize i = static_cast<usize>(blockIdx.x) * kBlock + tid;
    const usize stride = static_cast<usize>(gridDim.x) * kBlock;

    float local = 0.f;
    for (; i < n; i += stride) {
        const float v = fabsf(bf16_to_f32(W[i]));
        if (v > local) local = v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float other = __shfl_down_sync(0xffffffff, local, off);
        if (other > local) local = other;
    }
    if (lane == 0) warp_max[warp] = local;
    __syncthreads();
    if (warp == 0) {
        local = (tid < kBlock / 32) ? warp_max[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1) {
            const float other = __shfl_down_sync(0xffffffff, local, off);
            if (other > local) local = other;
        }
        if (lane == 0) atomicMax(reinterpret_cast<int*>(out), __float_as_int(local));
    }
}

template <class Fmt>
__global__ void cast_per_channel(
    const bf16* __restrict__ W,
    typename Fmt::store* __restrict__ out,
    const float* __restrict__ scale_inv,
    i32 cols) {
    const float s = scale_inv[blockIdx.x];
    const float s_recip = (s > 0.f) ? (1.f / s) : 0.f;
    const usize row_off = static_cast<usize>(blockIdx.x) * cols;
    for (i32 j = threadIdx.x; j < cols; j += blockDim.x) {
        out[row_off + j] = Fmt::narrow(bf16_to_f32(W[row_off + j]) * s_recip);
    }
}


__global__ void quant_act_fp8_per_group(
    const bf16* __restrict__ act,
    u8* __restrict__ out,
    float* __restrict__ scale_out,
    i32 m,
    i32 k,
    i32 gs,
    i32 n_groups) {
    const i32 row = blockIdx.y;
    const i32 g = blockIdx.x;
    if (row >= m || g >= n_groups) return;

    const i32 base = g * gs;
    const i32 remaining = k - base;
    const i32 count = (gs < remaining) ? gs : remaining;
    const usize off = static_cast<usize>(row) * k + base;

    float amax = 0.f;
    for (i32 i = threadIdx.x; i < count; i += blockDim.x) {
        amax = fmaxf(amax, fabsf(bf16_to_f32(act[off + i])));
    }
    __shared__ float warp_max[128 / 32];
    const unsigned lane = threadIdx.x & 31;
    const unsigned warp = threadIdx.x / 32;
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, o));
    }
    if (lane == 0) warp_max[warp] = amax;
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = warp_max[0];
        for (unsigned w = 1; w < blockDim.x / 32; ++w) v = fmaxf(v, warp_max[w]);
        warp_max[0] = v;
    }
    __syncthreads();
    amax = warp_max[0];

    const float scale = (amax > 0.f) ? (amax / fp8_e4m3::max_abs()) : 1.f;
    const float scale_rcp = (amax > 0.f) ? (fp8_e4m3::max_abs() / amax) : 0.f;
    if (threadIdx.x == 0) {
        scale_out[static_cast<usize>(row) * n_groups + g] = scale;
    }
    for (i32 i = threadIdx.x; i < count; i += blockDim.x) {
        out[off + i] = fp8_e4m3::narrow(bf16_to_f32(act[off + i]) * scale_rcp);
    }
}




// **THE TWO ENCODE KERNELS STOOD HERE** — `quant_per_channel` (bf16 to fp8
// e4m3, one inverse scale a row) and `quant_bf16_to_mxfp4_row` (e2m1 codes in
// 32-element blocks under one e8m0 exponent), with the `encode_fp4_e2m1` and
// `encode_e8m0` codepoint helpers they were the only readers of. §M-3 shut the
// load-time door they were the device half of: a serving load does not
// quantize, and `pie model import` writes the codes on the host, where
// `checkpoint::codec::mxfp4` holds the same two functions and is now the only
// statement of them.

__device__ __constant__ float kFp4Lut[16] = {
     0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
    -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f,
};

__device__ __forceinline__ void mxfp4_unpack8(unsigned word, __half2 out[4]) {
    constexpr unsigned kMagHi01234567 = 0x3E3C3800u;
    constexpr unsigned kMagHi4567     = 0x46444240u;
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

__device__ __forceinline__ float mxfp4_block_scale(u8 b) {
    return b == 0 ? exp2f(-127.f)
                  : __int_as_float(static_cast<int>(b) << 23);
}

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

        const float scale = exp2f(static_cast<float>(static_cast<int>(e8m0)) - 127.f);

        const int packed_base = blk * 16;
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

    constexpr int kPairs = kPairsT;
    constexpr int kRows = 2 * kPairs;
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

    const int words_per_row = hidden / 8;
    const int groups_per_row = hidden / 32;

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
        const auto* ub = up_bias_ptrs != nullptr
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
            const auto* ub = up_bias_ptrs != nullptr
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

// **WHERE A PACKED GROUP'S TWO PLANES ARE** (alto streaming §3 item 3, wave
// B7) — the packed path's answer to `moe_expert_base`, one granularity up.
//
// A split-plane bank is CODES beside SCALES, both indexed by the same expert
// id, and the select below computes both bases itself. That arithmetic is why
// the unit of residency on this path is the GROUP and not the expert (W-5's
// finding, `experts.rs`' header), and until this wave it was also why a group
// placed at load could never move: the two plane addresses were kernel
// PARAMETERS, and a captured graph holds its parameters forever (article 7).
//
// So the group gains one cell of DATA at a fixed device address, holding the
// two addresses the launch used to carry. A promotion writes the cell; the
// captured graph is untouched; the next fire reads the new tier.
//
// **THE PAIR IS ONE WORD, WHICH IS WHY A TORN PAIR STAYS UNCONSTRUCTIBLE.**
// Both plane addresses live in ONE 16-byte, 16-byte-aligned cell, so there is
// no cell state that names one group's codes and another's scales — the shell
// writes the pair as a unit, and this reads it as a unit: `ld.global.v2.u64`,
// **one extra load per group per launch**, issued once by every thread of a
// grid that all read the same address, so it is one L1 broadcast and not one
// load per route or per row.
//
// `nullptr` is the fully-resident load, and it is not the slow arm of a fast
// one: the cell is not read at all and the bases are the kernel parameters
// they always were. Same branch shape as `moe_expert_base`, same reason.
struct alignas(16) MoeGroupBases {
    const u8* codes;
    const u8* scales;
    // The affine planes' third base — null for the two-plane mxfp4 pair,
    // whose kernel never reads it. The pad keeps the cell a whole number of
    // 16-byte words so a cell write is two aligned stores.
    const u8* biases;
    const u8* pad;
};

// **THE ONE STATISTIC THE PACKED PATH PUBLISHES** — one `atomicAdd` per ROUTE
// per launch, by the one block that owns the first row tile of that route, so
// the count is "routed rows through this group" and not "blocks launched".
// The dense path counts per EXPERT (`moe_note_expert`); this one counts per
// GROUP, because the group is what can move.
//
// `nullptr` — the fully-resident load — costs the uniform branch and nothing.
__device__ __forceinline__ void moe_note_group(
    unsigned int* __restrict__ group_hits)
{
    if (group_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0) {
        atomicAdd(group_hits, 1u);
    }
}

// The routed matmul over an mxfp4 bank. `bias` is optional: a bank-cut leg
// hands its per-expert row and the add lands inside the fold, a rows-cut one
// hands nullptr and lets the routed bias mixture be stated after the reduce.
//
// `bases` and `group_hits` are the streamed seat, both `nullptr` for a
// resident bank — see `MoeGroupBases`. `win` is the staged-geometry seat,
// read in ROUTE space off a pair written in TOKEN space, and `top_k` is the
// fan-out that converts between them.
template <class T, int kRowsT>
__global__ void moe_matmul_select_mxfp4(
    const T* __restrict__ act,
    const i32* __restrict__ routes,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const T* __restrict__ bias,
    T* __restrict__ out,
    int top_k,
    int act_div,
    int n,
    int k,
    const MoeGroupBases* __restrict__ bases,
    unsigned int* __restrict__ group_hits,
    const u32* __restrict__ win)
{
    constexpr int kRows = kRowsT;
    const int route = blockIdx.x;
    // **THE SEAT IS IN TOKEN ROWS AND THIS AXIS IS IN ROUTES**, and the
    // conversion between them is the fan-out — `moe_matmul_select_gemv_body`'s
    // idiom in `moe.cuh`, which states it first and on the same route axis. A
    // window of `win[0]` token rows starting at `win[1]` is a run of
    // `win[0] * top_k` routes starting at route `win[1] * top_k`. Multiply
    // once, and the routes, activation and result planes below all read a
    // route ordinal that is the PLANE's and not the launch's — which is what
    // `engine_cuda::SHIFTED` promises for a name on its list.
    if (win != nullptr && route >= static_cast<int>(win[0]) * top_k) return;
    const int plane_route = win != nullptr
        ? route + static_cast<int>(win[1]) * top_k
        : route;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 = (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kRows;
    if (row0 >= n) return;
    const int expert = routes[plane_route];

    moe_note_group(group_hits);

    const int groups_per_row = k / 32;
    const int words_per_row = k / 8;

    // THE ONE EXTRA LOAD. Sixteen bytes, one address for the whole grid, and
    // the pair is read as a pair — see `MoeGroupBases`.
    const u8* codes_at = codes;
    const u8* scales_at = scales;
    if (bases != nullptr) {
        const MoeGroupBases seat = *bases;
        codes_at = seat.codes;
        scales_at = seat.scales;
    }

    const u8* w = codes_at + static_cast<long long>(expert) * n * (k / 2);
    const u8* s = scales_at + static_cast<long long>(expert) * n * groups_per_row;
    // The activation follows the same ordinal into whichever space it was
    // cut in: `act_div` is the fan-out on the up leg, where `act` holds one
    // row per token, and one on the down leg, where it holds one per route.
    const T* x = act + static_cast<long long>(plane_route / act_div) * k;

    int row_of[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) row_of[r] = min(row0 + r, n - 1);

    const unsigned* w32 = reinterpret_cast<const unsigned*>(w);

    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.f;

    for (int g = lane_id; g < groups_per_row; g += 32) {

        float part[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) part[r] = 0.f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            float xv[8];
#pragma unroll
            for (int j = 0; j < 8; ++j)
                xv[j] = Elem<T>::to_f32(x[g * 32 + q * 8 + j]);
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                __half2 qd[4];
                mxfp4_unpack8(
                    w32[static_cast<long long>(row_of[r]) * words_per_row + g * 4 + q],
                    qd);
#pragma unroll
                for (int j = 0; j < 4; ++j) {

                    const float2 f = __half22float2(qd[j]);
                    part[r] = fmaf(f.x, xv[2 * j], part[r]);
                    part[r] = fmaf(f.y, xv[2 * j + 1], part[r]);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            acc[r] = fmaf(
                part[r],
                mxfp4_block_scale(
                    s[static_cast<long long>(row_of[r]) * groups_per_row + g]),
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
        const T* b = bias != nullptr
            ? bias + static_cast<long long>(expert) * n
            : nullptr;
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const int row = row0 + r;
            if (row >= n) break;
            float v = acc[r];
            if (b != nullptr) v += Elem<T>::to_f32(b[row]);
            out[static_cast<long long>(plane_route) * n + row] = Elem<T>::from_f32(v);
        }
    }
}

// **THE AFFINE TWIN** — MLX's 4-bit codes, eight to a `u32` word, sixty-four
// under one bf16 scale and one bf16 zero point (`code * scale + bias`). The
// zero point folds through the group's activation sum, so each activation is
// read once: `Σ (c·s + b)·x = s·Σ c·x + b·Σ x`.
//
// Same grid as the mxfp4 select: one route per block-x, `kRowsT` bank rows
// per warp, a lane per group striding the row's groups. `bases` and
// `group_hits` are the streamed seat, both `nullptr` for a resident bank —
// and the base cell's THIRD pointer is this kernel's, see `MoeGroupBases`.
// The staged-geometry seat is the twin's too, `top_k` and all.
template <class T, int kRowsT>
__global__ void moe_matmul_select_mlxu4(
    const T* __restrict__ act,
    const i32* __restrict__ routes,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const u8* __restrict__ biases,
    T* __restrict__ out,
    int top_k,
    int act_div,
    int n,
    int k,
    const MoeGroupBases* __restrict__ bases,
    unsigned int* __restrict__ group_hits,
    const u32* __restrict__ win)
{
    constexpr int kRows = kRowsT;
    const int route = blockIdx.x;
    // The staged-geometry seat, in ROUTE space off a pair written in TOKEN
    // space: `moe_matmul_select_mxfp4`'s conversion above, same grid, same
    // fan-out (`moe.cuh` states it first).
    if (win != nullptr && route >= static_cast<int>(win[0]) * top_k) return;
    const int plane_route = win != nullptr
        ? route + static_cast<int>(win[1]) * top_k
        : route;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 = (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kRows;
    if (row0 >= n) return;
    const int expert = routes[plane_route];

    moe_note_group(group_hits);

    const int groups_per_row = k / 64;
    const int words_per_row = k / 8;

    const u8* codes_at = codes;
    const u8* scales_at = scales;
    const u8* biases_at = biases;
    if (bases != nullptr) {
        const MoeGroupBases seat = *bases;
        codes_at = seat.codes;
        scales_at = seat.scales;
        biases_at = seat.biases;
    }

    const unsigned* w32 = reinterpret_cast<const unsigned*>(
        codes_at + static_cast<long long>(expert) * n * (k / 2));
    const bf16* s16 = reinterpret_cast<const bf16*>(
        scales_at + static_cast<long long>(expert) * n * groups_per_row * 2);
    const bf16* b16 = reinterpret_cast<const bf16*>(
        biases_at + static_cast<long long>(expert) * n * groups_per_row * 2);
    // The activation follows the same ordinal into whichever space it was
    // cut in: `act_div` is the fan-out on the up leg, where `act` holds one
    // row per token, and one on the down leg, where it holds one per route.
    const T* x = act + static_cast<long long>(plane_route / act_div) * k;

    int row_of[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) row_of[r] = min(row0 + r, n - 1);

    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.f;

    for (int g = lane_id; g < groups_per_row; g += 32) {

        float part[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) part[r] = 0.f;
        float xsum = 0.f;

#pragma unroll
        for (int q = 0; q < 8; ++q) {
            float xv[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                xv[j] = Elem<T>::to_f32(x[g * 64 + q * 8 + j]);
                xsum += xv[j];
            }
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                const unsigned word =
                    w32[static_cast<long long>(row_of[r]) * words_per_row + g * 8 + q];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float code = static_cast<float>((word >> (4 * j)) & 0xFu);
                    part[r] = fmaf(code, xv[j], part[r]);
                }
            }
        }
        // xsum accumulated once per q-pass above counts every activation of
        // the group exactly once across the eight words.
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const long long fx =
                static_cast<long long>(row_of[r]) * groups_per_row + g;
            const float sv = Elem<bf16>::to_f32(s16[fx]);
            const float bv = Elem<bf16>::to_f32(b16[fx]);
            acc[r] = fmaf(part[r], sv, acc[r]);
            acc[r] = fmaf(xsum, bv, acc[r]);
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int r = 0; r < kRows; ++r)
            acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
    }
    if (lane_id == 0) {
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const int row = row0 + r;
            if (row < n)
                out[static_cast<long long>(plane_route) * n + row] =
                    Elem<T>::from_f32(acc[r]);
        }
    }
}

/// **THE OFFSET ARM** an affine projection spends its `xsum` on —
/// `matmul_affine`'s `kOffset` axis, and the launch side's `OffsetKind`.
///
/// `dtype::quant::OffSub` is the algebra these project. Three are its arms
/// literally (`Post(L(f))`, `Pre(L(U(b)))`, `Pre(L(f))`); the fourth is the
/// one that is NOT an offset in the signature at all — a symmetric term over
/// excess-binary codes (`Leaf::I(b)`), whose `c − 2^(b−1)` decode IS a
/// constant pre-offset once it reaches a dot. Q4_0, Q8_0 and Int4B8 land
/// there after canon, which is why there is no zero-offset arm below: a
/// point that only ever multiplied would be `linear::nvfp4`'s, not this one.
constexpr int kOffPost = 0;
constexpr int kOffPreInt = 1;
constexpr int kOffPreReal = 2;
constexpr int kOffPreConst = 3;

// **THE DENSE AFFINE FAMILY, ON ONE SKELETON** (qwen4 stored-form wave,
// generalized by QNF P1): `linear.matmul` and `linear.lm_head` over a weight
// the store seats as codes plus one factor per group of them — MLX's affine
// triplet, GPTQ/AWQ, HQQ, and the excess-binary symmetric rows, folded by
// one point rather than four.
//
// The identity is `moe_matmul_select_mlxu4`'s on an unrouted rectangle.
// Every arm accumulates the SAME pair per group — `part = Σ c·x` and
// `xsum = Σ x` — and they differ only in the epilogue that spends them:
//
//     kOffPost      s·part + b·xsum              an offset in the VALUE domain
//     kOffPreInt    s·(part − z·xsum)            an integer zero in the CODE domain
//     kOffPreReal   s·(part − z·xsum)            the same fold, `z` a real
//     kOffPreConst  s·(part − 2^(kBits−1)·xsum)  the zero the FORMAT fixes
//
// so each activation is read once and the factors land once per group. The
// `biases` plane is the offset's, and its bytes are the arm's: `kFactor`
// reals for `kOffPost` and `kOffPreReal`, ONE BYTE PER GROUP holding the
// unsigned code-domain zero for `kOffPreInt`, and nothing at all for
// `kOffPreConst`, which is fired with a null there.
//
// The GROUP is a runtime argument and not a constant: it comes off the
// factor plane's own width at launch, so thirty-two, sixty-four and a
// hundred and twenty-eight all fold here, and the entry refuses a width that
// groups a row into nothing whole. What stays constant is that a group is a
// whole number of code WORDS — eight codes at four bits, four at eight.
//
// One block column per ACTIVATION ROW (`blockIdx.x`), which is the decode
// shape: a step's row count is small and the weight is read once per row.
// A long prefill re-reads the weight per row through this grid — the tiled
// point that amortises it is deliberately not here yet; it arrives with a
// caller that measures it (the first-light prefills are single-digit rows).
//
// `bases` is the streamed seat, `nullptr` for a resident plane — see
// `MoeGroupBases`. No hit counter: a dense plane is not a routed group,
// and the tier does not note it (`engine_cuda::experts` D2b).
template <class T, class F, int kBits, int kOffset, int kGroup, int kRowsT>
__global__ void matmul_affine(
    const T* __restrict__ act,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const u8* __restrict__ biases,
    T* __restrict__ out,
    int n,
    int k,
    const MoeGroupBases* __restrict__ bases,
    const u32* __restrict__ win)
{
    constexpr int kRows = kRowsT;
    constexpr int kPerWord = 32 / kBits;
    constexpr unsigned kMask = (1u << kBits) - 1u;
    // The excess-binary midpoint, the only offset this point holds itself.
    constexpr float kExcess = static_cast<float>(1 << (kBits - 1));
    const int token = blockIdx.x;
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && token >= static_cast<int>(win[0])) return;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 = (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kRows;
    if (row0 >= n) return;

    const u8* codes_at = codes;
    const u8* scales_at = scales;
    const u8* biases_at = biases;
    if (bases != nullptr) {
        const MoeGroupBases seat = *bases;
        codes_at = seat.codes;
        scales_at = seat.scales;
        biases_at = seat.biases;
    }

    // The group width is a template argument so this loop's bound is a
    // constant the compiler unrolls — the decode path lost a quarter of its
    // tokens/s when the bound went runtime, and the jit's name-expression
    // cache only ever holds the (bits, group) pairs a model actually fires.
    const int groups_per_row = k / kGroup;
    constexpr int kWordsPerGroup = kGroup / kPerWord;
    const int words_per_row = k / kPerWord;
    const unsigned* w32 = reinterpret_cast<const unsigned*>(codes_at);
    const F* sf = reinterpret_cast<const F*>(scales_at);
    // The offset plane under the two readings an arm may take of it: a real
    // beside the scale, or one unsigned code-domain zero per group.
    const F* bf = reinterpret_cast<const F*>(biases_at);
    const u8* zb = biases_at;
    const T* x = act + static_cast<long long>(token) * k;

    int row_of[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) row_of[r] = min(row0 + r, n - 1);

    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.f;

    for (int g = lane_id; g < groups_per_row; g += 32) {

        float part[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) part[r] = 0.f;
        float xsum = 0.f;

#pragma unroll
        for (int q = 0; q < kWordsPerGroup; ++q) {
            float xv[kPerWord];
#pragma unroll
            for (int j = 0; j < kPerWord; ++j) {
                xv[j] = Elem<T>::to_f32(x[g * kGroup + q * kPerWord + j]);
                xsum += xv[j];
            }
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                const unsigned word =
                    w32[static_cast<long long>(row_of[r]) * words_per_row
                        + g * kWordsPerGroup + q];
#pragma unroll
                for (int j = 0; j < kPerWord; ++j) {
                    const float code =
                        static_cast<float>((word >> (kBits * j)) & kMask);
                    part[r] = fmaf(code, xv[j], part[r]);
                }
            }
        }
        // xsum accumulated once per q-pass above counts every activation of
        // the group exactly once across the group's words.
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const long long fx =
                static_cast<long long>(row_of[r]) * groups_per_row + g;
            const float sv = Elem<F>::to_f32(sf[fx]);
            if constexpr (kOffset == kOffPost) {
                acc[r] = fmaf(part[r], sv, acc[r]);
                acc[r] = fmaf(xsum, Elem<F>::to_f32(bf[fx]), acc[r]);
            } else {
                // One fold for the three code-domain arms: they agree on
                // `s·(part − z·xsum)` and disagree only on where `z` is.
                float z;
                if constexpr (kOffset == kOffPreInt) {
                    z = static_cast<float>(zb[fx]);
                } else if constexpr (kOffset == kOffPreReal) {
                    z = Elem<F>::to_f32(bf[fx]);
                } else {
                    z = kExcess;
                }
                acc[r] = fmaf(sv, fmaf(-z, xsum, part[r]), acc[r]);
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int r = 0; r < kRows; ++r)
            acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
    }
    if (lane_id == 0) {
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const int row = row0 + r;
            if (row < n)
                out[static_cast<long long>(token) * n + row] =
                    Elem<T>::from_f32(acc[r]);
        }
    }
}

// **THE SAME WEIGHT, DECODED INSTEAD OF FOLDED** — the INTERIM prefill arm
// of the dense affine family (`linear::quant::matmul_via_dense`).
//
// `matmul_affine` above gives one block column per ACTIVATION ROW and reads
// the whole weight inside each of them. At one token that is parity with
// cuBLAS; at a prefill's hundreds it is the same weight read hundreds of
// times, measured at 98-189x cuBLAS bf16 over 128..2048 rows. So at prefill
// shapes the caller decodes the weight ONCE into a transient scratch tile
// and fires the dense point on it. **The stored form does not change**: this
// is a fire-time buffer, and the row the store seats is still codes plus
// factors.
//
// The element written is `matmul_affine`'s epilogue with the activation
// taken back out of it. That epilogue accumulates
// `s·Σc·x + b·Σx = Σ(s·c + b)·x` under `kOffPost` and
// `s·(Σc·x − z·Σx) = Σ s·(c − z)·x` under the three code-domain arms, so
// what a decoded weight element IS, arm for arm, is:
//
//     kOffPost      s·c + b
//     kOffPreInt    s·(c − z)     `z` one unsigned byte per group
//     kOffPreReal   s·(c − z)     `z` a factor-dtype real
//     kOffPreConst  s·(c − 2^(kBits−1))
//
// — the same planes, the same `fx` indexing, the same constant midpoint.
//
// **bf16, ROW-MAJOR `[n, k]`**, which is the rectangle `linear::gemm`'s
// `act x w^T` reads. The decode rounds each element to bf16 exactly once,
// which is the whole numeric difference between this arm and the fused one:
// they answer the same numbers, not the same bits.
//
// **ONE THREAD PER CODE WORD** — eight elements at four bits, four at eight
// — flat over `n · k / kPerWord`, because this point is bandwidth and not
// arithmetic: one word and two factors in, `kPerWord` halves out.
//
// **NO WIN GUARD, DELIBERATELY.** `matmul_affine`'s `win` word retires the
// padded rows of a grid carved over TOKEN rows; this grid is carved over the
// WEIGHT's rows, which no bucket pads and no replay reshapes. And no `bases`
// seat either: the launch side refuses a streamed seat, because a plane that
// moves between fires has no fixed rectangle to decode into a slab.
template <class F, int kBits, int kOffset, int kGroup>
__global__ void dequant_affine(
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const u8* __restrict__ biases,
    bf16* __restrict__ out,
    int n,
    int k)
{
    constexpr int kPerWord = 32 / kBits;
    constexpr unsigned kMask = (1u << kBits) - 1u;
    // The excess-binary midpoint, this point's only self-held offset — the
    // same constant `matmul_affine` folds.
    constexpr float kExcess = static_cast<float>(1 << (kBits - 1));

    const int words_per_row = k / kPerWord;
    const long long words = static_cast<long long>(n) * words_per_row;
    const long long at =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= words) return;

    const int row = static_cast<int>(at / words_per_row);
    const int word_in_row = static_cast<int>(at % words_per_row);
    // A group is a whole number of code words (the entry refuses anything
    // else), so a word belongs to exactly one group and this division is the
    // group it belongs to.
    const int groups_per_row = k / kGroup;
    const long long fx = static_cast<long long>(row) * groups_per_row
        + (word_in_row * kPerWord) / kGroup;

    // The offset plane under the two readings an arm may take of it, and
    // under the one arm that reads nothing at all.
    const F* sf = reinterpret_cast<const F*>(scales);
    const F* bf = reinterpret_cast<const F*>(biases);
    const u8* zb = biases;
    const float sv = Elem<F>::to_f32(sf[fx]);
    float off;
    if constexpr (kOffset == kOffPost || kOffset == kOffPreReal) {
        off = Elem<F>::to_f32(bf[fx]);
    } else if constexpr (kOffset == kOffPreInt) {
        off = static_cast<float>(zb[fx]);
    } else {
        off = kExcess;
    }

    const unsigned word = reinterpret_cast<const unsigned*>(codes)[at];
    bf16* dst = out + static_cast<long long>(row) * k + word_in_row * kPerWord;
#pragma unroll
    for (int j = 0; j < kPerWord; ++j) {
        const float code = static_cast<float>((word >> (kBits * j)) & kMask);
        float v;
        if constexpr (kOffset == kOffPost) {
            v = fmaf(code, sv, off);
        } else {
            v = sv * (code - off);
        }
        dst[j] = f32_to_bf16(v);
    }
}

template <class T>
__global__ void dequant_int8_per_channel(
    const i8* __restrict__ W,
    T* __restrict__ out,
    const float* __restrict__ scale_inv,
    i32 cols,
    usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const i32 row = static_cast<i32>(i / static_cast<usize>(cols));
    out[i] = Elem<T>::from_f32(static_cast<float>(W[i]) * scale_inv[row]);
}

__global__ void w8a8_dequant(
    const i32* __restrict__ acc,
    const float* __restrict__ act_inv,
    const float* __restrict__ w_inv,
    bf16* __restrict__ out,
    i32 M,
    i32 N) {
    const i32 n = blockIdx.x * blockDim.x + threadIdx.x;
    const i32 m = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= N || m >= M) return;
    const float v = static_cast<float>(acc[m * N + n]) * act_inv[m] * w_inv[n];
    out[m * N + n] = f32_to_bf16(v);
}

using f32 = float;

template <class T>
struct Cast {
    static __device__ __forceinline__ float to_f32(T v) { return Elem<T>::to_f32(v); }
    static __device__ __forceinline__ T from_f32(float v) { return Elem<T>::from_f32(v); }
};

template <>
struct Cast<f32> {
    static __device__ __forceinline__ float to_f32(f32 v) { return v; }
    static __device__ __forceinline__ f32 from_f32(float v) { return v; }
};

template <class T>
__global__ void cast_f32_to(const float* __restrict__ src, T* __restrict__ dst, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = Cast<T>::from_f32(src[i]);
}

template <class T>
__global__ void cast_to_f32(const T* __restrict__ src, float* __restrict__ dst, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = Cast<T>::to_f32(src[i]);
}

template <class T>
__global__ void cast_f16_to(const f16* __restrict__ src, T* __restrict__ dst, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = Cast<T>::from_f32(Cast<f16>::to_f32(src[i]));
}

template <class T>
__global__ void cast_e8m0_to(const u8* __restrict__ src, T* __restrict__ dst, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u32 bits = static_cast<u32>(src[i]);
    const float v = bits == 0xFFu ? __int_as_float(0x7FFFFFFF) : __int_as_float(bits << 23);
    dst[i] = Cast<T>::from_f32(v);
}

template <class T>
__global__ void scale(
    const T* __restrict__ src, T* __restrict__ dst, usize n, float factor) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = Cast<T>::from_f32(Cast<T>::to_f32(src[i]) * factor);
}

template <class T>
__global__ void scale_rows(T* buf, const T* l, int width) {
    const int row = blockIdx.x;
    T* row_buf = buf + static_cast<usize>(row) * width;
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
        row_buf[c] =
            Cast<T>::from_f32(Cast<T>::to_f32(row_buf[c]) * Cast<T>::to_f32(l[c]));
    }
}

__global__ void marlin_permute_scales_per_group(bf16* __restrict__ s, int total64_rows) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= total64_rows || tid >= 64) return;
    bf16* base = s + static_cast<usize>(row) * 64;
    __shared__ bf16 buf[64];
    buf[tid] = base[tid];
    __syncthreads();
    const int i = tid / 8;
    const int j = tid % 8;
    const int src_idx = j * 8 + i;
    base[tid] = buf[src_idx];
}

__global__ void awq_dequant_to_bf16(
    const u32* __restrict__ qweight,
    const u32* __restrict__ qzeros,
    const bf16* __restrict__ scales,
    bf16* __restrict__ out,
    int size_k,
    int size_n,
    int group_size) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= size_n || k >= size_k) return;

    constexpr int REV[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    const int n8 = size_n / 8;
    const int n_packed = n / 8;
    const int n_in_8 = n % 8;
    const int shift = 4 * REV[n_in_8];

    const int g = k / group_size;
    const u32 w_word = qweight[k * n8 + n_packed];
    const u32 zp_word = qzeros[g * n8 + n_packed];
    const int w_int4 = static_cast<int>((w_word >> shift) & 0xFu);
    const int zp_int4 = static_cast<int>((zp_word >> shift) & 0xFu);

    const float sc = bf16_to_f32(scales[g * size_n + n]);
    const float val = static_cast<float>(w_int4 - zp_int4) * sc;
    out[n * size_k + k] = f32_to_bf16(val);
}

__global__ void gptq_dequant_to_bf16(
    const u32* __restrict__ qweight,
    const u32* __restrict__ qzeros,
    const bf16* __restrict__ scales,
    const i32* __restrict__ g_idx,
    bf16* __restrict__ out,
    int size_k,
    int size_n,
    int group_size) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= size_n || k >= size_k) return;

    const int n8 = size_n / 8;
    const int g = (g_idx != nullptr) ? g_idx[k] : (k / group_size);

    const u32 w_word = qweight[(k / 8) * size_n + n];
    const u32 z_word = qzeros[g * n8 + (n / 8)];
    const int w_int4 = static_cast<int>((w_word >> ((k % 8) * 4)) & 0xFu);
    const int zp_int4 = static_cast<int>((z_word >> ((n % 8) * 4)) & 0xFu) + 1;

    const float sc = bf16_to_f32(scales[g * size_n + n]);
    const float val = static_cast<float>(w_int4 - zp_int4) * sc;
    out[n * size_k + k] = f32_to_bf16(val);
}

}
