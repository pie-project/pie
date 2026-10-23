#include "linear/act_quant.cuh"
#include "prim/device.cuh"
#ifdef __CUDACC_RTC__

#include "prim/mma_sm80.cuh"
#else

#include <cuda_bf16.h>
#include <mma.h>
#endif

namespace pie::linear {

// A word's two bf16 as f32 (`lo` low).
__device__ __forceinline__ float bf16x2_lo(u32 w) { return __uint_as_float(w << 16); }

__device__ __forceinline__ float bf16x2_hi(u32 w) { return __uint_as_float(w & 0xffff0000u); }

// Two bf16 products rounded once (an fma adding -0, which keeps a zero's
// sign).
__device__ __forceinline__ u32 bf16x2_mul(u32 a, u32 b)
{
    u32 w;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(w) : "r"(a), "r"(b), "r"(0x80008000u));
    return w;
}

// The block's sum of each of `v`'s K values, as norm_gate's block folds one
// (warps, then the warps' sums in the first), all K at once; `part` holds
// 33 * K and is reusable on return.
template <int K>
__device__ __forceinline__ void route_row_sums(float (&v)[K], float* part)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        v[k] = ::pie::warp_sum(v[k]);
        if (lane == 0) part[warp * K + k] = v[k];
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int k = 0; k < K; ++k) {
            float w = lane < static_cast<int>(blockDim.x >> 5) ? part[lane * K + k] : 0.f;
            w = ::pie::warp_sum(w);
            if (lane == 0) part[32 * K + k] = w;
        }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < K; ++k) v[k] = part[32 * K + k];
    __syncthreads();
}

// `moe_weighted_sum_vec8` of the grouped product read where it lies: token
// t's eight columns from `grouped[pos[t * top + j]]`, weighted, summed in f32
// in `j` order; a dropped route adds nothing. A token past the live window
// (`win`, in tokens) is untouched. Normed, a block holds the row (`n / 8`
// threads) and each route is first normed as norm_gate norms it under
// `scale`: `u = x * scale` rounded, `u / rms(u)` rounded (a row of at most
// 4096). `kAheadNorm` normed routes are in flight at once (each normed by its
// own block sum, the same whatever the batch).
template <bool kNorm, int kAheadNorm = 2>
__global__ void __launch_bounds__(kNorm ? 512 : 1024, kNorm ? (kAheadNorm > 2 ? 1 : 4) : 1) route_weighted_gather_vec8(
    const bf16* __restrict__ grouped,
    const i32* __restrict__ pos,
    const float* __restrict__ weights,
    const bf16* __restrict__ shared,
    const bf16* __restrict__ gate,
    int gate_pitch,
    bf16* __restrict__ y,
    int n,
    int top,
    const bf16* __restrict__ scale,
    float eps,
    const u32* __restrict__ win)
{
    const int t = blockIdx.x;
    if (win != nullptr && t >= static_cast<int>(win[0])) return;
    const int plane = win != nullptr ? t + static_cast<int>(win[1]) : t;
    const int h0 = (blockIdx.y * blockDim.x + threadIdx.x) * 8;
    if (h0 >= n) return;
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.f;
    // Eight routes' rows (and weights) in flight at once, summed in route
    // order: a row's latency paid once a batch, not once a route. Normed,
    // `kAheadNorm`, their norms' block sums taken together: two lets a row's
    // block fit four to a multiprocessor, which hides more latency than a
    // deeper batch over a long fire; a decode's few rows' blocks leave the
    // multiprocessors idle, and a deeper batch pays fewer rounds.
    constexpr int kAhead = kNorm ? kAheadNorm : 8;
    __shared__ float part[kNorm ? 33 * kAhead : 1];
    uint4 sv = {};
    if constexpr (kNorm) sv = *reinterpret_cast<const uint4*>(scale + h0);
    const long long base = static_cast<long long>(plane) * top;
    for (int j0 = 0; j0 < top; j0 += kAhead) {
        int at[kAhead];
        uint4 rows[kAhead];
        float wts[kAhead];
#pragma unroll
        for (int a = 0; a < kAhead; ++a) at[a] = j0 + a < top ? pos[base + j0 + a] : -1;
#pragma unroll
        for (int a = 0; a < kAhead; ++a) {
            rows[a] = at[a] >= 0 ? *reinterpret_cast<const uint4*>(grouped + static_cast<long long>(at[a]) * n + h0)
                                 : uint4{};
            // No weights: the routes already carry theirs (DeepSeek's, before
            // the down leg), and moe_sum_add sums them as they are.
            wts[a] = weights != nullptr && at[a] >= 0 ? weights[base + j0 + a] : 1.f;
        }
        if constexpr (kNorm) {
            // Two values a word: `x * scale` rounded is one bf16 product
            // (the f32 one is exact), and `u / rms(u)` rounds in pairs.
            float inv[kAhead];
#pragma unroll
            for (int a = 0; a < kAhead; ++a) {
                u32* r = reinterpret_cast<u32*>(&rows[a]);
                const u32* sc = reinterpret_cast<const u32*>(&sv);
                inv[a] = 0.f;
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    r[w] = bf16x2_mul(r[w], sc[w]);
                    inv[a] = fmaf(bf16x2_lo(r[w]), bf16x2_lo(r[w]), inv[a]);
                    inv[a] = fmaf(bf16x2_hi(r[w]), bf16x2_hi(r[w]), inv[a]);
                }
            }
            route_row_sums(inv, part);
#pragma unroll
            for (int a = 0; a < kAhead; ++a) {
                if (at[a] < 0) continue;
                const float v = rsqrtf(inv[a] / static_cast<float>(n) + eps), wr = wts[a];
                const u32* r = reinterpret_cast<const u32*>(&rows[a]);
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const u32 o = pack_bf16x2(bf16x2_lo(r[w]) * v, bf16x2_hi(r[w]) * v);
                    acc[2 * w] += wr * bf16x2_lo(o);
                    acc[2 * w + 1] += wr * bf16x2_hi(o);
                }
            }
        } else {
#pragma unroll
            for (int a = 0; a < kAhead; ++a) {
                if (at[a] < 0) continue;
                const bf16* v8 = reinterpret_cast<const bf16*>(&rows[a]);
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[i] += wts[a] * bf16_to_f32(v8[i]);
            }
        }
    }
    // Qwen's shared expert under its sigmoid gate, as moe_sigmoid_gate_add
    // adds it; with no gate, the shared row added as it is (moe_sum_add).
    if (shared != nullptr) {
        float s = 1.f;
        if (gate != nullptr) {
            const float gv = bf16_to_f32(gate[static_cast<long long>(plane) * gate_pitch]);
            s = 1.f / (1.f + __expf(-gv));
        }
        const uint4 sv = *reinterpret_cast<const uint4*>(shared + static_cast<long long>(plane) * n + h0);
        const bf16* s8 = reinterpret_cast<const bf16*>(&sv);
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] += bf16_to_f32(s8[i]) * s;
    }
    uint4 out;
    bf16* o8 = reinterpret_cast<bf16*>(&out);
#pragma unroll
    for (int i = 0; i < 8; ++i) o8[i] = f32_to_bf16(acc[i]);
    *reinterpret_cast<uint4*>(y + static_cast<long long>(plane) * n + h0) = out;
}

// `out[pos[r]] = v[r]`: a route-order f32 a route landed at its grouped row
// (a dropped route nowhere).
__global__ void route_scatter_f32(
    const float* __restrict__ v,
    const i32* __restrict__ pos,
    float* __restrict__ out,
    int count)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= count) return;
    const int o = pos[r];
    if (o >= 0) out[o] = v[r];
}

// `grouped[pos[r]] = x[r / div]`, 16 bytes a thread; a dropped route lands
// nowhere.
__global__ void route_scatter_rows(
    const bf16* __restrict__ x,
    const i32* __restrict__ pos,
    bf16* __restrict__ grouped,
    int count,
    int k,
    int div)
{
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int runs = k / 8;
    if (at >= static_cast<long long>(count) * runs) return;
    const int r = static_cast<int>(at / runs);
    const int c = static_cast<int>(at % runs) * 8;
    const int o = pos[r];
    if (o < 0) return;
    *reinterpret_cast<uint4*>(grouped + static_cast<long long>(o) * k + c) =
        *reinterpret_cast<const uint4*>(x + static_cast<long long>(r / div) * k + c);
}

template <class T>
__global__ void moe_sigmoid_gate_add(
    T* __restrict__ out,
    const T* __restrict__ sum,
    const T* __restrict__ x,
    const T* __restrict__ scalar_gate,
    i32 H, i32 stride,
    const u32* __restrict__ win)
{
    const i32 n = blockIdx.x;

    if (win != nullptr && n >= static_cast<i32>(win[0])) return;

    const i32 plane_row = win != nullptr ? n + static_cast<i32>(win[1]) : n;
    const i32 h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= H) return;

    const float gv =
        Elem<T>::to_f32(scalar_gate[static_cast<long long>(plane_row) * stride]);
    const float s = 1.f / (1.f + __expf(-gv));
    const long long i = static_cast<long long>(plane_row) * H + h;
    const float ov = Elem<T>::to_f32(sum[i]);
    const float xv = Elem<T>::to_f32(x[i]);
    out[i] = Elem<T>::from_f32(ov + xv * s);
}

template <class T>
__global__ void moe_weighted_sum(
    T* __restrict__ out,
    const T* __restrict__ src,
    const float* __restrict__ weights,
    int top_k, int hidden,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int plane_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= hidden) return;

    const long long base = static_cast<long long>(plane_row) * top_k;
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < top_k; ++k) {
        const long long r = base + k;
        const float v = Elem<T>::to_f32(src[r * hidden + h]);
        acc += weights[r] * v;
    }
    out[static_cast<long long>(plane_row) * hidden + h] = Elem<T>::from_f32(acc);
}

// `moe_weighted_sum` eight columns a thread, 16 bytes at a time: the same
// sum, in the same order, for every element.
template <class T>
__global__ void moe_weighted_sum_vec8(
    T* __restrict__ out,
    const T* __restrict__ src,
    const float* __restrict__ weights,
    int top_k, int hidden,
    const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int plane_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int h0 = (blockIdx.y * blockDim.x + threadIdx.x) * 8;
    if (h0 >= hidden) return;

    const long long base = static_cast<long long>(plane_row) * top_k;
    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.f;
    for (int k = 0; k < top_k; ++k) {
        const long long r = base + k;
        const uint4 w = *reinterpret_cast<const uint4*>(src + r * hidden + h0);
        const T* v8 = reinterpret_cast<const T*>(&w);
        const float wr = weights[r];
        #pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] += wr * Elem<T>::to_f32(v8[i]);
    }
    uint4 o;
    T* o8 = reinterpret_cast<T*>(&o);
    #pragma unroll
    for (int i = 0; i < 8; ++i) o8[i] = Elem<T>::from_f32(acc[i]);
    *reinterpret_cast<uint4*>(out + static_cast<long long>(plane_row) * hidden + h0) = o;
}

// Each token's weighted routed rows plus its shared row, summed in f32.
template <class T>
__global__ void moe_sum_add(
    T* __restrict__ out,
    const T* __restrict__ src,
    const T* __restrict__ shared,
    int top_k, int hidden,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int plane_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= hidden) return;

    const long long base = static_cast<long long>(plane_row) * top_k;
    float acc = 0.f;
    for (int k = 0; k < top_k; ++k) {
        acc += Elem<T>::to_f32(src[(base + k) * hidden + h]);
    }
    const long long at = static_cast<long long>(plane_row) * hidden + h;
    out[at] = Elem<T>::from_f32(acc + Elem<T>::to_f32(shared[at]));
}

template <class T>
__global__ void moe_bias_sum(
    T* __restrict__ out,
    const T* __restrict__ x,
    const T* __restrict__ bias,
    const i32* __restrict__ topk_idx,
    const float* __restrict__ weights,
    int top_k, int hidden,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int plane_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= hidden) return;

    const long long base = static_cast<long long>(plane_row) * top_k;
    const long long i = static_cast<long long>(plane_row) * hidden + h;
    float acc = Elem<T>::to_f32(x[i]);
    for (int k = 0; k < top_k; ++k) {
        const long long r = base + k;
        const int e = topk_idx[r];

        if (e < 0) continue;
        const float b = Elem<T>::to_f32(
            bias[static_cast<long long>(e) * hidden + h]);
        acc += weights[r] * b;
    }
    out[i] = Elem<T>::from_f32(acc);
}

template <class T>
__global__ void reorder_moe_aligned_output(
    const T* __restrict__ aligned_out,
    const i32* __restrict__ sorted_route_ids,
    T* __restrict__ route_out,
    int num_routes,
    int aligned_rows,
    int hidden,
    int shared_row_begin,
    int num_tokens,
    T* __restrict__ shared_out)
{
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    const int row = blockIdx.x;
    if (h >= hidden || row >= aligned_rows) return;
    const T v = aligned_out[static_cast<long long>(row) * hidden + h];
    if (shared_row_begin >= 0 && row >= shared_row_begin) {
        const int t = row - shared_row_begin;
        if (t < num_tokens) {
            shared_out[static_cast<long long>(t) * hidden + h] = v;
        }
        return;
    }
    const int route = sorted_route_ids[row];
    if (route >= num_routes) return;
    route_out[static_cast<long long>(route) * hidden + h] = v;
}

}
