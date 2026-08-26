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

template <class Fmt>
__global__ void quant_per_channel(
    const bf16* __restrict__ W,
    typename Fmt::store* __restrict__ out,
    float* __restrict__ scale_inv,
    i32 cols) {
    const int tid = threadIdx.x;
    extern __shared__ float warp_max[];

    const usize row_off = static_cast<usize>(blockIdx.x) * cols;
    float local = 0.f;
    for (i32 j = tid; j < cols; j += kBlock) {
        const float v = fabsf(bf16_to_f32(W[row_off + j]));
        if (v > local) local = v;
    }
    const float row_max = row_absmax(local, warp_max, tid);

    const float quant = (row_max > 0.f) ? (Fmt::max_abs() / row_max) : 1.f;
    const float weight_scale_inv = (row_max > 0.f) ? (row_max / Fmt::max_abs()) : 1.f;
    if (tid == 0) scale_inv[blockIdx.x] = weight_scale_inv;

    for (i32 j = tid; j < cols; j += kBlock) {
        out[row_off + j] = Fmt::narrow(bf16_to_f32(W[row_off + j]) * quant);
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

__device__ __forceinline__ unsigned encode_fp4_e2m1(float x) {
    const float a = fabsf(x);
    unsigned mag;
    if (a < 0.25f) {
        mag = 0;
    } else if (a < 0.75f) {
        mag = 1;
    } else if (a < 1.25f) {
        mag = 2;
    } else if (a < 1.75f) {
        mag = 3;
    } else if (a < 2.5f) {
        mag = 4;
    } else if (a < 3.5f) {
        mag = 5;
    } else if (a < 5.0f) {
        mag = 6;
    } else {
        mag = 7;
    }
    const unsigned sign = (x < 0.0f) ? 0x8u : 0x0u;
    return (mag == 0) ? 0u : (sign | mag);
}

__device__ __forceinline__ u8 encode_e8m0(float absmax) {
    if (!(absmax > 0.0f)) return 0;
    const float l = log2f(absmax / 6.0f);
    int b = static_cast<int>(ceilf(l)) + 127;
    if (b < 0) b = 0;
    if (b > 254) b = 254;
    return static_cast<u8>(b);
}

template <class T>
__global__ void quant_bf16_to_mxfp4_row(
    const T* __restrict__ src,
    u8* __restrict__ packed,
    u8* __restrict__ scales,
    i32 cols) {
    const i32 row = blockIdx.x;
    const i32 groups = cols / 32;
    const usize row_src = static_cast<usize>(row) * cols;
    const usize row_packed = static_cast<usize>(row) * (cols / 2);
    const usize row_scale = static_cast<usize>(row) * groups;

    for (i32 g = threadIdx.x; g < groups; g += blockDim.x) {
        const i32 base = g * 32;
        float absmax = 0.0f;
        float vals[32];
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            const float v = Elem<T>::to_f32(src[row_src + base + k]);
            vals[k] = v;
            const float a = fabsf(v);
            if (a > absmax) absmax = a;
        }
        const u8 sb = encode_e8m0(absmax);
        scales[row_scale + g] = sb;

        const float s = ldexpf(1.0f, static_cast<int>(sb) - 127);
        const float inv_s = (s == 0.0f) ? 0.0f : (1.0f / s);
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const unsigned lo = encode_fp4_e2m1(vals[2 * k] * inv_s);
            const unsigned hi = encode_fp4_e2m1(vals[2 * k + 1] * inv_s);
            packed[row_packed + g * 16 + k] = static_cast<u8>((hi << 4) | (lo & 0xFu));
        }
    }
}

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

template <class T, int kRowsT>
__global__ void moe_matmul_select_bias_mxfp4(
    const T* __restrict__ act,
    const i32* __restrict__ routes,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const T* __restrict__ bias,
    T* __restrict__ out,
    int act_div,
    int n,
    int k)
{
    constexpr int kRows = kRowsT;
    const int route = blockIdx.x;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 = (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kRows;
    if (row0 >= n) return;
    const int expert = routes[route];

    const int groups_per_row = k / 32;
    const int words_per_row = k / 8;

    const u8* w = codes + static_cast<long long>(expert) * n * (k / 2);
    const u8* s = scales + static_cast<long long>(expert) * n * groups_per_row;
    const T* x = act + static_cast<long long>(route / act_div) * k;

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
            out[static_cast<long long>(route) * n + row] = Elem<T>::from_f32(v);
        }
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
