#pragma once

#include "prelude/device.cuh"

namespace pie::linear {

template <int kWarps, int kUnrollP = 4>
__global__ void gemv_bf16_kernel(
    const bf16* __restrict__ weight,
    const bf16* __restrict__ act,
    const bf16* __restrict__ bias,
    bf16* __restrict__ out,
    int N, int K, float beta)
{
    const int row = blockIdx.x * kWarps + threadIdx.y;
    if (row >= N) return;
    const float4* w4 =
        reinterpret_cast<const float4*>(weight + (long long)row * K);
    const float4* x4 = reinterpret_cast<const float4*>(act);
    const int vectors = K / 8;
    constexpr int kUnroll = kUnrollP;
    float acc = 0.f;
    int i = threadIdx.x;
    for (; i + 32 * (kUnroll - 1) < vectors; i += 32 * kUnroll) {
        float4 wv[kUnroll];
        float4 xv[kUnroll];
        #pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            wv[u] = w4[i + 32 * u];
            xv[u] = x4[i + 32 * u];
        }
        #pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const bf16* wb = reinterpret_cast<const bf16*>(&wv[u]);
            const bf16* xb = reinterpret_cast<const bf16*>(&xv[u]);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc += bf16_to_f32(wb[j]) * bf16_to_f32(xb[j]);
            }
        }
    }
    for (; i < vectors; i += 32) {
        float4 wv = w4[i];
        float4 xv = x4[i];
        const bf16* wb = reinterpret_cast<const bf16*>(&wv);
        const bf16* xb = reinterpret_cast<const bf16*>(&xv);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc += bf16_to_f32(wb[j]) * bf16_to_f32(xb[j]);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (threadIdx.x == 0) {
        if (beta != 0.f) acc += beta * bf16_to_f32(out[row]);

        bf16 v = f32_to_bf16(acc);
        if (bias != nullptr) {
            v = f32_to_bf16(bf16_to_f32(v) + bf16_to_f32(bias[row]));
        }
        out[row] = v;
    }
}

// `out[row][n] = act[row] . weight[n]` for a product a few columns wide
// (`N` <= 8): a block a row, its threads striding the contraction eight
// elements at a time with every column's sum in registers, folded across
// the block. The bias, if any, lands as the gemv lands it.
template <int N>
__global__ void __launch_bounds__(256) gemm_narrow_bf16_kernel(
    const bf16* __restrict__ weight,
    const bf16* __restrict__ act,
    const bf16* __restrict__ bias,
    bf16* __restrict__ out,
    int k)
{
    constexpr int kThreads = 256;
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const bf16* x = act + static_cast<long long>(row) * k;
    float acc[N];
#pragma unroll
    for (int n = 0; n < N; ++n) acc[n] = 0.f;
    for (int i = tid * 8; i < k; i += kThreads * 8) {
        const uint4 xv = *reinterpret_cast<const uint4*>(x + i);
        const bf16* x8 = reinterpret_cast<const bf16*>(&xv);
        float xf[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) xf[e] = bf16_to_f32(x8[e]);
#pragma unroll
        for (int n = 0; n < N; ++n) {
            const uint4 wv = *reinterpret_cast<const uint4*>(weight + static_cast<long long>(n) * k + i);
            const bf16* w8 = reinterpret_cast<const bf16*>(&wv);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[n] += xf[e] * bf16_to_f32(w8[e]);
        }
    }
    __shared__ float part[kThreads / 32][N];
#pragma unroll
    for (int n = 0; n < N; ++n) {
        float v = acc[n];
#pragma unroll
        for (int sh = 16; sh > 0; sh >>= 1) v += __shfl_xor_sync(0xffffffffu, v, sh);
        if ((tid & 31) == 0) part[tid >> 5][n] = v;
    }
    __syncthreads();
    if (tid < N) {
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < kThreads / 32; ++w) v += part[w][tid];
        bf16 o = f32_to_bf16(v);
        if (bias != nullptr) o = f32_to_bf16(bf16_to_f32(o) + bf16_to_f32(bias[tid]));
        out[static_cast<long long>(row) * N + tid] = o;
    }
}

template <int kWarps, int kUnrollP = 1>
__global__ void gemv_splitk_bf16_kernel(
    const bf16* __restrict__ weight,
    const bf16* __restrict__ act,
    const bf16* __restrict__ bias,
    bf16* __restrict__ out,
    int N, int K, float beta)
{
    const int row = blockIdx.x;
    if (row >= N) return;
    const int warp = threadIdx.y;
    const float4* w4 =
        reinterpret_cast<const float4*>(weight + (long long)row * K);
    const float4* x4 = reinterpret_cast<const float4*>(act);
    const int vectors = K / 8;

    float acc = 0.f;

    constexpr int kU = kUnrollP;
    const int stride = kWarps * 32;
    int i = warp * 32 + threadIdx.x;
    for (; i + stride * (kU - 1) < vectors; i += stride * kU) {
        float4 wv[kU];
        float4 xv[kU];
        #pragma unroll
        for (int u = 0; u < kU; ++u) {
            wv[u] = w4[i + stride * u];
            xv[u] = x4[i + stride * u];
        }
        #pragma unroll
        for (int u = 0; u < kU; ++u) {
            const bf16* wb = reinterpret_cast<const bf16*>(&wv[u]);
            const bf16* xb = reinterpret_cast<const bf16*>(&xv[u]);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc += bf16_to_f32(wb[j]) * bf16_to_f32(xb[j]);
            }
        }
    }
    for (; i < vectors; i += stride) {
        float4 wv = w4[i];
        float4 xv = x4[i];
        const bf16* wb = reinterpret_cast<const bf16*>(&wv);
        const bf16* xb = reinterpret_cast<const bf16*>(&xv);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc += bf16_to_f32(wb[j]) * bf16_to_f32(xb[j]);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }

    __shared__ float partial[kWarps];
    if (threadIdx.x == 0) partial[warp] = acc;
    __syncthreads();
    if (warp != 0 || threadIdx.x != 0) return;
    float total = 0.f;
    #pragma unroll
    for (int w = 0; w < kWarps; ++w) total += partial[w];
    if (beta != 0.f) total += beta * bf16_to_f32(out[row]);

    bf16 v = f32_to_bf16(total);
    if (bias != nullptr) {
        v = f32_to_bf16(bf16_to_f32(v) + bf16_to_f32(bias[row]));
    }
    out[row] = v;
}

}
