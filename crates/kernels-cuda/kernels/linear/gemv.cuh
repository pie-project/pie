#pragma once

#include "prelude/device.cuh"

namespace pie::linear {

// A warp's dot of weight row `row` (K wide) with the activation `act`,
// folded to lane 0.
template <int kUnroll>
__device__ __forceinline__ float gemv_bf16_row(
    const bf16* __restrict__ weight, const bf16* __restrict__ act, int row, int K)
{
    const float4* w4 =
        reinterpret_cast<const float4*>(weight + (long long)row * K);
    const float4* x4 = reinterpret_cast<const float4*>(act);
    const int vectors = K / 8;
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
    return acc;
}

template <int kWarps, int kUnrollP = 4>
__global__ void gemv_bf16_kernel(
    const bf16* __restrict__ weight,
    const bf16* __restrict__ act,
    const bf16* __restrict__ bias,
    bf16* __restrict__ out,
    int N, int K, float beta)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const int row = blockIdx.x * kWarps + threadIdx.y;
    if (row >= N) return;
    float acc = gemv_bf16_row<kUnrollP>(weight, act, row, K);
    if (threadIdx.x == 0) {
        if (beta != 0.f) acc += beta * bf16_to_f32(out[row]);

        bf16 v = f32_to_bf16(acc);
        if (bias != nullptr) {
            v = f32_to_bf16(bf16_to_f32(v) + bf16_to_f32(bias[row]));
        }
        out[row] = v;
    }
}

// `gemv_bf16_kernel` over a grouped plane at one row: weight row `row` is
// group `row / group_n`'s, against that group's K-wide slice of the row.
template <int kWarps>
__global__ void gemv_bf16_grouped_kernel(
    const bf16* __restrict__ weight,
    const bf16* __restrict__ act,
    bf16* __restrict__ out,
    int N, int K, int group_n)
{
    const int row = blockIdx.x * kWarps + threadIdx.y;
    if (row >= N) return;
    const float acc = gemv_bf16_row<4>(weight, act + (long long)(row / group_n) * K, row, K);
    if (threadIdx.x == 0) out[row] = f32_to_bf16(acc);
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
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
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

// `out[r][n] = act[r] . weight[n]` for a few rows (`kM` <= 8, a decode's):
// each of a block's eight warps streams four weight rows' slice of K, a
// weight vector of each against every activation row (read through L1, so
// each activation load serves four weight loads), `kSplit` warps the rows'
// slices; each warp's sums fold across its lanes, and a row's slices add in
// slice order, so a launch lands the same bits each time. One launch, every
// weight byte read once: a split GEMM's reduce launch saved. Over a cluster
// of `kCl` blocks (a few columns, too few blocks to keep the loads in
// flight), each block takes a slice of K and the first adds the blocks' sums
// in rank order.
template <int kM, int kSplit, int kCl = 1>
__global__ void __launch_bounds__(256) gemm_rows_bf16_kernel(
    const bf16* __restrict__ weight,
    const bf16* __restrict__ act,
    const bf16* __restrict__ bias,
    bf16* __restrict__ out,
    int N, int K)
{
    static_assert(kM >= 1 && kM <= 8, "a few rows");
    // A programmatic dependent (`kernel.pdl`) reads nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    static_assert(8 % kSplit == 0, "a block's warps split evenly over its rows");
    constexpr int kR = 2;                     // weight rows a warp
    constexpr int kRows = 8 / kSplit * kR;    // weight rows a block
    __shared__ float red[8][kR][kM];
    __shared__ float sum[kRows * kM];
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int block = static_cast<int>(blockIdx.x) / kCl;
    const int rank = static_cast<int>(blockIdx.x) % kCl;
    const int row0 = block * kRows + warp / kSplit * kR;
    const int part = rank * kSplit + warp % kSplit;
    const int vectors = K / 8;
    const int per = vectors / (kSplit * kCl);
    float acc[kR][kM];
#pragma unroll
    for (int q = 0; q < kR; ++q) {
#pragma unroll
        for (int r = 0; r < kM; ++r) acc[q][r] = 0.f;
    }
    const uint4* x4 = reinterpret_cast<const uint4*>(act) + part * per;
    const uint4* w4[kR];
#pragma unroll
    for (int q = 0; q < kR; ++q) {
        // A row past N reads row N - 1 again and lands nothing.
        const int row = min(row0 + q, N - 1);
        w4[q] = reinterpret_cast<const uint4*>(weight + static_cast<long long>(row) * K) + part * per;
    }
#pragma unroll 4
    for (int i = lane; i < per; i += 32) {
        uint4 wv[kR];
#pragma unroll
        for (int q = 0; q < kR; ++q) wv[q] = __ldcs(w4[q] + i);
#pragma unroll
        for (int r = 0; r < kM; ++r) {
            const uint4 xv = __ldg(x4 + static_cast<long long>(r) * vectors + i);
            const unsigned xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
            for (int q = 0; q < kR; ++q) {
                const unsigned ww[4] = {wv[q].x, wv[q].y, wv[q].z, wv[q].w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[q][r] = fmaf(__uint_as_float(ww[j] << 16), __uint_as_float(xw[j] << 16),
                                     acc[q][r]);
                    acc[q][r] = fmaf(__uint_as_float(ww[j] & 0xffff0000u),
                                     __uint_as_float(xw[j] & 0xffff0000u), acc[q][r]);
                }
            }
        }
    }
#pragma unroll
    for (int q = 0; q < kR; ++q) {
#pragma unroll
        for (int r = 0; r < kM; ++r) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                acc[q][r] += __shfl_xor_sync(0xffffffffu, acc[q][r], off);
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int q = 0; q < kR; ++q) {
#pragma unroll
            for (int r = 0; r < kM; ++r) red[warp][q][r] = acc[q][r];
        }
    }
    __syncthreads();
    // Thread t sums block row t / kM's column for activation row t % kM.
    for (int t = static_cast<int>(threadIdx.x); t < kRows * kM; t += 256) {
        const int local = t / kM;
        const int r = t % kM;
        const int group = local / kR;   // the warp group holding the row
        const int q = local % kR;
        float s = red[group * kSplit][q][r];
#pragma unroll
        for (int p = 1; p < kSplit; ++p) s += red[group * kSplit + p][q][r];
        sum[t] = s;
    }
    if constexpr (kCl > 1) {
        asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    } else {
        __syncthreads();
    }
    if (rank == 0) {
        for (int t = static_cast<int>(threadIdx.x); t < kRows * kM; t += 256) {
            const int n = block * kRows + t / kM;
            float s = sum[t];
#pragma unroll
            for (int c = 1; c < kCl; ++c) {
                const unsigned local = static_cast<unsigned>(__cvta_generic_to_shared(sum + t));
                unsigned remote;
                float v;
                asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local), "r"(c));
                asm("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(remote));
                s += v;
            }
            if (n < N) {
                bf16 v = f32_to_bf16(s);
                if (bias != nullptr) {
                    v = f32_to_bf16(bf16_to_f32(v) + bf16_to_f32(bias[n]));
                }
                out[static_cast<long long>(t % kM) * N + n] = v;
            }
        }
    }
    if constexpr (kCl > 1) {
        // No block retires while the first may read its sums.
        asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
}

}
