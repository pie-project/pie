#pragma once

#include "prelude/device.cuh"

namespace pie::elemwise {

template <int BLOCK>
__device__ __forceinline__ float block_reduce_sum_exact(float local, float* buf)
{
    static_assert(BLOCK >= 32 && (BLOCK & (BLOCK - 1)) == 0,
                  "block_reduce_sum_exact needs a power-of-two BLOCK >= 32");
    const int tid = threadIdx.x;
    buf[tid] = local;
    __syncthreads();
#pragma unroll
    for (int off = BLOCK / 2; off >= 32; off >>= 1) {
        if (tid < off) buf[tid] += buf[tid + off];
        __syncthreads();
    }
    if (tid < 32) {
        float v = buf[tid];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, off);
        }
        if (tid == 0) buf[0] = v;
    }
    __syncthreads();
    return buf[0];
}

template <int BLOCK>
__device__ __forceinline__ float block_reduce_sum_fast(float local, float* buf)
{
    static_assert(BLOCK >= 32 && BLOCK <= 1024 && (BLOCK & (BLOCK - 1)) == 0,
                  "block_reduce_sum_fast needs a power-of-two BLOCK in [32, 1024]");
    constexpr int kWarps = BLOCK / 32;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local += __shfl_xor_sync(0xffffffffu, local, off);
    }
    if (lane == 0) buf[warp] = local;
    __syncthreads();
    if (warp == 0) {
        float v = tid < kWarps ? buf[tid] : 0.f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_xor_sync(0xffffffffu, v, off);
        }
        if (lane == 0) buf[kWarps] = v;
    }
    __syncthreads();
    return buf[kWarps];
}

template <class T, int BLOCK, bool WEIGHT_PLUS_ONE>
__device__ __forceinline__ void rmsnorm_row(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int hidden,
    int x_row_stride,
    int y_row_stride,
    float eps,
    int heads,
    const u32* __restrict__ win)
{
    const int row = blockIdx.x;

    if (win != nullptr && row / heads >= static_cast<int>(win[0])) return;

    const int plane_row =
        win != nullptr ? row + static_cast<int>(win[1]) * heads : row;

    const int tid = threadIdx.x;

    const T* xr = x + static_cast<long long>(plane_row) * x_row_stride;
    T* yr = y + static_cast<long long>(plane_row) * y_row_stride;

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float v = Elem<T>::to_f32(xr[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_fast<BLOCK>(local, buf);

    const float inv_rms = rsqrtf(buf_sum / static_cast<float>(hidden) + eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        const float xv = Elem<T>::to_f32(xr[i]);
        float wv = Elem<T>::to_f32(weight[i]);
        if constexpr (WEIGHT_PLUS_ONE) wv += 1.f;
        yr[i] = Elem<T>::from_f32(xv * inv_rms * wv);
    }
}

template <class T, int BLOCK = 256>
__global__ void rmsnorm(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int hidden,
    int x_row_stride,
    int y_row_stride,
    float eps,
    int heads,
    const u32* __restrict__ win)
{
    rmsnorm_row<T, BLOCK, false>(
        x, weight, y, hidden, x_row_stride, y_row_stride, eps, heads, win);
}

template <class T, int BLOCK = 256>
__global__ void rmsnorm_plus_one(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int hidden,
    int x_row_stride,
    int y_row_stride,
    float eps,
    int heads,
    const u32* __restrict__ win)
{
    rmsnorm_row<T, BLOCK, true>(
        x, weight, y, hidden, x_row_stride, y_row_stride, eps, heads, win);
}

// A wide row's RMS norm (its width a whole number of eights, at most
// `256 * 8 * kRuns`) a block a row: each thread's 16-byte runs loaded at
// once and held through the block's sum, so the row is read once.
template <class T, int kRuns, bool WEIGHT_PLUS_ONE>
__global__ void __launch_bounds__(256) rmsnorm_runs(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int hidden,
    int x_row_stride,
    int y_row_stride,
    float eps,
    const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const int row = blockIdx.x;
    if (win != nullptr && row >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? row + static_cast<int>(win[1]) : row;
    const int tid = threadIdx.x;
    const T* xr = x + static_cast<long long>(plane_row) * x_row_stride;
    T* yr = y + static_cast<long long>(plane_row) * y_row_stride;
    const int runs = hidden >> 3;
    uint4 v[kRuns];
#pragma unroll
    for (int r = 0; r < kRuns; ++r) {
        const int at = tid + r * 256;
        v[r] = at < runs ? *reinterpret_cast<const uint4*>(xr + at * 8) : make_uint4(0, 0, 0, 0);
    }
    float local = 0.f;
#pragma unroll
    for (int r = 0; r < kRuns; ++r) {
        const T* e = reinterpret_cast<const T*>(&v[r]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float f = Elem<T>::to_f32(e[j]);
            local += f * f;
        }
    }
    __shared__ float buf[256];
    const float sum = block_reduce_sum_fast<256>(local, buf);
    const float inv_rms = rsqrtf(sum / static_cast<float>(hidden) + eps);
#pragma unroll
    for (int r = 0; r < kRuns; ++r) {
        const int at = tid + r * 256;
        if (at >= runs) continue;
        const uint4 wr = *reinterpret_cast<const uint4*>(weight + at * 8);
        const T* e = reinterpret_cast<const T*>(&v[r]);
        const T* wv = reinterpret_cast<const T*>(&wr);
        uint4 o;
        T* oe = reinterpret_cast<T*>(&o);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float w = Elem<T>::to_f32(wv[j]);
            if constexpr (WEIGHT_PLUS_ONE) w += 1.f;
            oe[j] = Elem<T>::from_f32(Elem<T>::to_f32(e[j]) * inv_rms * w);
        }
        *reinterpret_cast<uint4*>(yr + at * 8) = o;
    }
}

// `rmsnorm` (`rmsnorm_plus_one` under PLUS_ONE) of a whole row of `256 *
// PER`, landing its bits: each thread's values (its elements `t`, `t + 256`,
// ...) and their weights loaded at once and held, the squares summed in the
// same order, not read twice in a loop each.
template <class T, int PER, bool PLUS_ONE>
__global__ void __launch_bounds__(256) rmsnorm_held(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ y,
    float eps,
    const u32* __restrict__ win)
{
    constexpr int BLOCK = 256;
    constexpr int hidden = BLOCK * PER;
    const int row = blockIdx.x;
    if (win != nullptr && row >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? row + static_cast<int>(win[1]) : row;
    const int tid = threadIdx.x;
    const T* xr = x + static_cast<long long>(plane_row) * hidden;
    T* yr = y + static_cast<long long>(plane_row) * hidden;
    float v[PER], w[PER];
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        v[k] = Elem<T>::to_f32(xr[tid + k * BLOCK]);
        w[k] = Elem<T>::to_f32(weight[tid + k * BLOCK]);
    }
    float local = 0.f;
#pragma unroll
    for (int k = 0; k < PER; ++k) local += v[k] * v[k];
    __shared__ float buf[BLOCK];
    const float inv_rms = rsqrtf(block_reduce_sum_fast<BLOCK>(local, buf) / static_cast<float>(hidden) + eps);
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        float wv = w[k];
        if constexpr (PLUS_ONE) wv += 1.f;
        yr[tid + k * BLOCK] = Elem<T>::from_f32(v[k] * inv_rms * wv);
    }
}

template <int PER> struct LaneRun { using type = uint4; };
template <> struct LaneRun<4> { using type = uint2; };

// A 128- or 256-wide RMS norm one warp per (row, head): a lane carries
// `PER` consecutive values in one 8- or 16-byte load, so the sum of squares
// is one butterfly and no block barrier; a block is eight such units.
// `units` counts (row, head) pairs; the rows are dense.
template <class T, int PER, bool WEIGHT_PLUS_ONE>
__global__ void __launch_bounds__(256) rmsnorm_in_warp(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int units,
    float eps,
    int heads,
    const u32* __restrict__ win)
{
    static_assert(PER == 4 || PER == 8, "a lane's run is one 8- or 16-byte load");
    constexpr int hidden = 32 * PER;
    using Run = typename LaneRun<PER>::type;
    const int unit = static_cast<int>(blockIdx.x) * 8 + static_cast<int>(threadIdx.x >> 5);
    if (unit >= units) return;
    if (win != nullptr && unit / heads >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? unit + static_cast<int>(win[1]) * heads : unit;
    const int lane = threadIdx.x & 31;
    const long long at = static_cast<long long>(plane_row) * hidden + lane * PER;

    T xv[PER];
    T wv[PER];
    *reinterpret_cast<Run*>(xv) = *reinterpret_cast<const Run*>(x + at);
    *reinterpret_cast<Run*>(wv) = *reinterpret_cast<const Run*>(weight + lane * PER);
    float xf[PER];
    float local = 0.f;
    #pragma unroll
    for (int u = 0; u < PER; ++u) {
        xf[u] = Elem<T>::to_f32(xv[u]);
        local += xf[u] * xf[u];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) local += __shfl_xor_sync(0xffffffffu, local, off);
    const float inv_rms = rsqrtf(local / static_cast<float>(hidden) + eps);

    T out[PER];
    #pragma unroll
    for (int u = 0; u < PER; ++u) {
        float w = Elem<T>::to_f32(wv[u]);
        if constexpr (WEIGHT_PLUS_ONE) w += 1.f;
        out[u] = Elem<T>::from_f32(xf[u] * inv_rms * w);
    }
    *reinterpret_cast<Run*>(y + at) = *reinterpret_cast<const Run*>(out);
}

// Each `group`-wide run of a row RMS-normed and scaled by its own weights,
// plus one under `PLUS_ONE`.
template <class T, int BLOCK, bool PLUS_ONE>
__global__ void rmsnorm_grouped(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int group,
    int groups,
    float eps,
    const u32* __restrict__ win)
{
    const int b = blockIdx.x;

    if (win != nullptr && b / groups >= static_cast<int>(win[0])) return;

    const int gb = win != nullptr ? b + static_cast<int>(win[1]) * groups : b;

    const int tid = threadIdx.x;

    const T* xr = x + static_cast<long long>(gb) * group;
    const T* wr = weight + static_cast<long long>(b % groups) * group;
    T* yr = y + static_cast<long long>(gb) * group;

    float local = 0.f;
    for (int i = tid; i < group; i += BLOCK) {
        const float v = Elem<T>::to_f32(xr[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_fast<BLOCK>(local, buf);
    const float inv_rms = rsqrtf(buf_sum / static_cast<float>(group) + eps);

    for (int i = tid; i < group; i += BLOCK) {
        const float xv = Elem<T>::to_f32(xr[i]);
        const float wv = Elem<T>::to_f32(wr[i]) + (PLUS_ONE ? 1.f : 0.f);
        yr[i] = Elem<T>::from_f32(xv * inv_rms * wv);
    }
}

template <int BLOCK, bool WEIGHT_PLUS_ONE, bool EMIT_FP16 = false>
__global__ void rmsnorm_vec8(
    const bf16* __restrict__ x,
    const bf16* __restrict__ weight,
    bf16* __restrict__ y,

    f16* __restrict__ y_fp16,
    int hidden,
    int x_row_stride,
    int y_row_stride,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nvec = hidden / 8;

    const float4* xr =
        reinterpret_cast<const float4*>(x + static_cast<long long>(row) * x_row_stride);
    float4* yr =
        reinterpret_cast<float4*>(y + static_cast<long long>(row) * y_row_stride);
    const float4* wr = reinterpret_cast<const float4*>(weight);

    float local = 0.f;
    for (int i = tid; i < nvec; i += BLOCK) {
        float4 v = xr[i];
        const bf16x2* h = reinterpret_cast<const bf16x2*>(&v);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = bf16x2_to_f32(h[j]);
            local += f.x * f.x + f.y * f.y;
        }
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_fast<BLOCK>(local, buf);
    const float inv_rms = rsqrtf(buf_sum / static_cast<float>(hidden) + eps);

    for (int i = tid; i < nvec; i += BLOCK) {
        float4 v = xr[i];
        float4 g = wr[i];
        float4 o;
        const bf16x2* hv = reinterpret_cast<const bf16x2*>(&v);
        const bf16x2* hg = reinterpret_cast<const bf16x2*>(&g);
        bf16x2* ho = reinterpret_cast<bf16x2*>(&o);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 a = bf16x2_to_f32(hv[j]);
            float2 b = bf16x2_to_f32(hg[j]);
            if constexpr (WEIGHT_PLUS_ONE) { b.x += 1.f; b.y += 1.f; }
            ho[j] = f32_to_bf16x2(a.x * inv_rms * b.x,
                                  a.y * inv_rms * b.y);
        }
        yr[i] = o;
        if constexpr (EMIT_FP16) {

            const bf16* ob = reinterpret_cast<const bf16*>(&o);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                y_fp16[i * 8 + j] = f32_to_f16(bf16_to_f32(ob[j]));
            }
        }
    }
}

template <class T, int BLOCK, bool WEIGHT_PLUS_ONE>
__global__ void residual_add_rmsnorm(
    const T* __restrict__ x,
    T* __restrict__ y,
    const T* __restrict__ weight,
    T* __restrict__ out,
    int hidden,
    float eps,
    const u32* __restrict__ win)
{
    const int row = blockIdx.x;

    if (win != nullptr && row >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? row + static_cast<int>(win[1]) : row;
    const int tid = threadIdx.x;

    const T* xr = x + static_cast<long long>(plane_row) * hidden;
    T* yr = y + static_cast<long long>(plane_row) * hidden;
    T* outr = out + static_cast<long long>(plane_row) * hidden;

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const T summed = Elem<T>::from_f32(Elem<T>::to_f32(yr[i]) + Elem<T>::to_f32(xr[i]));
        yr[i] = summed;
        const float v = Elem<T>::to_f32(summed);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_fast<BLOCK>(local, buf);
    const float inv_rms = rsqrtf(buf_sum / static_cast<float>(hidden) + eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        const float xv = Elem<T>::to_f32(yr[i]);
        float wv = Elem<T>::to_f32(weight[i]);
        if constexpr (WEIGHT_PLUS_ONE) wv += 1.f;
        outr[i] = Elem<T>::from_f32(xv * inv_rms * wv);
    }
}

template <int BLOCK, bool WEIGHT_PLUS_ONE>
__global__ void residual_add_rmsnorm_vec8(
    const bf16* __restrict__ x,
    bf16* __restrict__ y,
    const bf16* __restrict__ weight,
    bf16* __restrict__ out,
    int hidden,
    float eps,
    const u32* __restrict__ win)
{
    const int row = blockIdx.x;
    if (win != nullptr && row >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? row + static_cast<int>(win[1]) : row;
    const int tid = threadIdx.x;
    const int nvec = hidden / 8;
    const long long base = static_cast<long long>(plane_row) * hidden;

    const uint4* xr = reinterpret_cast<const uint4*>(x + base);
    uint4* yr = reinterpret_cast<uint4*>(y + base);
    uint4* outr = reinterpret_cast<uint4*>(out + base);
    const uint4* wr = reinterpret_cast<const uint4*>(weight);

    float local = 0.f;
    for (int i = tid; i < nvec; i += BLOCK) {
        uint4 yv = yr[i];
        const uint4 xv = xr[i];
        bf16x2* yh = reinterpret_cast<bf16x2*>(&yv);
        const bf16x2* xh = reinterpret_cast<const bf16x2*>(&xv);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 a = bf16x2_to_f32(yh[j]);
            const float2 b = bf16x2_to_f32(xh[j]);
            yh[j] = f32_to_bf16x2(a.x + b.x, a.y + b.y);
            const float2 f = bf16x2_to_f32(yh[j]);
            local += f.x * f.x + f.y * f.y;
        }
        yr[i] = yv;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_fast<BLOCK>(local, buf);
    const float inv_rms = rsqrtf(buf_sum / static_cast<float>(hidden) + eps);

    for (int i = tid; i < nvec; i += BLOCK) {
        const uint4 yv = yr[i];
        const uint4 wv = wr[i];
        uint4 ov;
        const bf16x2* yh = reinterpret_cast<const bf16x2*>(&yv);
        const bf16x2* wh = reinterpret_cast<const bf16x2*>(&wv);
        bf16x2* oh = reinterpret_cast<bf16x2*>(&ov);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 v = bf16x2_to_f32(yh[j]);
            float2 w = bf16x2_to_f32(wh[j]);
            if constexpr (WEIGHT_PLUS_ONE) {
                w.x += 1.f;
                w.y += 1.f;
            }
            oh[j] = f32_to_bf16x2(v.x * inv_rms * w.x, v.y * inv_rms * w.y);
        }
        outr[i] = ov;
    }
}

// residual_add_rmsnorm_vec8 in one pass: a thread keeps its PER vectors of
// the summed row (and its weight vectors) in registers across the block's
// reduction, all its loads issued up front, so the normed row is written
// without reading the stream back. The sums run in the same order, so it
// lands the same bits.
template <int PER, bool WEIGHT_PLUS_ONE>
__global__ void __launch_bounds__(256) residual_add_rmsnorm_reg(
    const bf16* __restrict__ x,
    bf16* __restrict__ y,
    const bf16* __restrict__ weight,
    bf16* __restrict__ out,
    int hidden,
    float eps,
    const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    constexpr int BLOCK = 256;
    const int row = blockIdx.x;
    if (win != nullptr && row >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? row + static_cast<int>(win[1]) : row;
    const int tid = threadIdx.x;
    const int nvec = hidden / 8;
    const long long base = static_cast<long long>(plane_row) * hidden;

    const uint4* xr = reinterpret_cast<const uint4*>(x + base);
    uint4* yr = reinterpret_cast<uint4*>(y + base);
    uint4* outr = reinterpret_cast<uint4*>(out + base);
    const uint4* wr = reinterpret_cast<const uint4*>(weight);

    uint4 yv[PER], xv[PER], wv[PER];
    #pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int i = tid + p * BLOCK;
        if (i < nvec) {
            yv[p] = yr[i];
            xv[p] = xr[i];
            wv[p] = wr[i];
        }
    }
    float local = 0.f;
    #pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int i = tid + p * BLOCK;
        if (i < nvec) {
            bf16x2* yh = reinterpret_cast<bf16x2*>(&yv[p]);
            const bf16x2* xh = reinterpret_cast<const bf16x2*>(&xv[p]);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 a = bf16x2_to_f32(yh[j]);
                const float2 b = bf16x2_to_f32(xh[j]);
                yh[j] = f32_to_bf16x2(a.x + b.x, a.y + b.y);
                const float2 f = bf16x2_to_f32(yh[j]);
                local += f.x * f.x + f.y * f.y;
            }
            yr[i] = yv[p];
        }
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_fast<BLOCK>(local, buf);
    const float inv_rms = rsqrtf(buf_sum / static_cast<float>(hidden) + eps);

    #pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int i = tid + p * BLOCK;
        if (i < nvec) {
            uint4 ov;
            const bf16x2* yh = reinterpret_cast<const bf16x2*>(&yv[p]);
            const bf16x2* wh = reinterpret_cast<const bf16x2*>(&wv[p]);
            bf16x2* oh = reinterpret_cast<bf16x2*>(&ov);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 v = bf16x2_to_f32(yh[j]);
                float2 w = bf16x2_to_f32(wh[j]);
                if constexpr (WEIGHT_PLUS_ONE) {
                    w.x += 1.f;
                    w.y += 1.f;
                }
                oh[j] = f32_to_bf16x2(v.x * inv_rms * w.x, v.y * inv_rms * w.y);
            }
            outr[i] = ov;
        }
    }
}

template <class T, int BLOCK = 256>
__global__ void rmsnorm_no_scale(
    const T* __restrict__ x,
    T* __restrict__ y,
    int hidden,
    float eps,
    int heads,
    const u32* __restrict__ win)
{
    const int row = blockIdx.x;

    if (win != nullptr && row / heads >= static_cast<int>(win[0])) return;

    const int plane_row =
        win != nullptr ? row + static_cast<int>(win[1]) * heads : row;

    const int tid = threadIdx.x;

    const T* xr = x + static_cast<long long>(plane_row) * hidden;
    T* yr = y + static_cast<long long>(plane_row) * hidden;

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float v = Elem<T>::to_f32(xr[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_fast<BLOCK>(local, buf);
    const float inv_rms = rsqrtf(buf_sum / static_cast<float>(hidden) + eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        yr[i] = Elem<T>::from_f32(Elem<T>::to_f32(xr[i]) * inv_rms);
    }
}

// A norm_gate row's tail, `u` its (scaled) values and `rv` its residual's:
// the norm, the gated sum (`gate_of(e, alpha, beta)` a channel's gate), `y`,
// and under NEXT the next norm into `z`.
template <class T, int BLOCK, bool NEXT, class Gate>
__device__ __forceinline__ void norm_gate_out(
    const float (&u)[8],
    const T (&rv)[8],
    bool gated,
    Gate gate_of,
    long long base,
    int at,
    T* __restrict__ y,
    float eps,
    const T* __restrict__ next_w,
    float next_eps,
    int plus_one,
    T* __restrict__ z,
    float* buf)
{
    constexpr int hidden = 8 * BLOCK;
    float local = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) local = fmaf(u[e], u[e], local);
    const float inv_rms = rsqrtf(block_reduce_sum_fast<BLOCK>(local, buf) / static_cast<float>(hidden) + eps);
    T out[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        if (!gated) {
            out[e] = Elem<T>::from_f32(u[e] * inv_rms);
        } else {
            const float n = Elem<T>::to_f32(Elem<T>::from_f32(u[e] * inv_rms));
            float alpha, beta;
            gate_of(e, alpha, beta);
            out[e] = Elem<T>::from_f32(alpha * Elem<T>::to_f32(rv[e]) + beta * n);
        }
    }
    *reinterpret_cast<uint4*>(y + base) = *reinterpret_cast<const uint4*>(out);
    if constexpr (NEXT) {
        T wt[8];
        *reinterpret_cast<uint4*>(wt) = *reinterpret_cast<const uint4*>(next_w + at);
        float sq = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float o = Elem<T>::to_f32(out[e]);
            sq += o * o;
        }
        __syncthreads();
        const float inv = rsqrtf(block_reduce_sum_fast<BLOCK>(sq, buf) / static_cast<float>(hidden) + next_eps);
        T zn[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            float wv = Elem<T>::to_f32(wt[e]);
            if (plus_one) wv += 1.f;
            zn[e] = Elem<T>::from_f32(Elem<T>::to_f32(out[e]) * inv * wv);
        }
        *reinterpret_cast<uint4*>(z + base) = *reinterpret_cast<const uint4*>(zn);
    }
}

// A gate's alpha and beta from its channel's `g`.
__device__ __forceinline__ void norm_gate_ab(float g, float temperature, float& alpha, float& beta)
{
    const float w = __fdividef(g, temperature);
    beta = __fdividef(1.f, 1.f + __expf(-w));
    const float a2 = fmaxf(__fdividef(1.f, 1.f + __expf(w)) * (1.f + beta), 1e-3f);
    alpha = a2 * rsqrtf(a2);
}

// `u = x * scale` rounded to T (`x` with no scale).
template <class T>
__device__ __forceinline__ void norm_gate_in(const T (&v)[8], const T* __restrict__ scale, int at, float (&u)[8])
{
    if (scale != nullptr) {
        T s[8];
        *reinterpret_cast<uint4*>(s) = *reinterpret_cast<const uint4*>(scale + at);
#pragma unroll
        for (int e = 0; e < 8; ++e)
            u[e] = Elem<T>::to_f32(Elem<T>::from_f32(Elem<T>::to_f32(v[e]) * Elem<T>::to_f32(s[e])));
    } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) u[e] = Elem<T>::to_f32(v[e]);
    }
}

// Muse Spark's post-norm and residual gate, a block a row of `8 * BLOCK`, a
// 16-byte run a thread held in registers: `u = x * scale` rounded to T (`x`
// with no scale), `n = u / rms(u)` rounded to T, and with a residual `y =
// alpha * res + beta * n` in f32 (`w = gate / temperature`, `beta =
// sigmoid(w)`, `alpha = sqrt(max(sigmoid(-w) * (1 + beta), 1e-3))`), else
// `y = n`. Under `NEXT` the next sublayer's input norm over `y` too: `z =
// rmsnorm(y) * w` (`1 + w` where `plus_one`).
template <class T, int BLOCK, bool NEXT>
__global__ void __launch_bounds__(BLOCK) norm_gate(
    const T* __restrict__ x,
    const T* __restrict__ scale,
    const T* __restrict__ res,
    const float* __restrict__ gate,
    T* __restrict__ y,
    float eps,
    float temperature,
    int fan,
    const T* __restrict__ next_w,
    float next_eps,
    int plus_one,
    T* __restrict__ z,
    const u32* __restrict__ win)
{
    constexpr int hidden = 8 * BLOCK;
    // A routed plane holds `fan` rows a token; the window counts tokens.
    const int row = blockIdx.x;
    if (win != nullptr && row >= static_cast<int>(win[0]) * fan) return;
    const int plane_row = win != nullptr ? row + static_cast<int>(win[1]) * fan : row;
    const int at = threadIdx.x * 8;
    const long long base = static_cast<long long>(plane_row) * hidden + at;
    T v[8];
    *reinterpret_cast<uint4*>(v) = *reinterpret_cast<const uint4*>(x + base);
    float u[8];
    norm_gate_in(v, scale, at, u);
    T rv[8];
    float g[8];
    if (res != nullptr) {
        *reinterpret_cast<uint4*>(rv) = *reinterpret_cast<const uint4*>(res + base);
        *reinterpret_cast<float4*>(g) = *reinterpret_cast<const float4*>(gate + at);
        *reinterpret_cast<float4*>(g + 4) = *reinterpret_cast<const float4*>(gate + at + 4);
    }
    __shared__ float buf[BLOCK];
    norm_gate_out<T, BLOCK, NEXT>(
        u, rv, res != nullptr, [&](int e, float& a, float& b) { norm_gate_ab(g[e], temperature, a, b); },
        base, at, y, eps, next_w, next_eps, plus_one, z, buf);
}

// norm_gate over a long fire, a block ROWS rows in turn: its channels' gate
// math taken once, and each row's loads out while the last is normed.
template <class T, int BLOCK, bool NEXT, int ROWS>
__global__ void __launch_bounds__(BLOCK) norm_gate_rows(
    const T* __restrict__ x,
    const T* __restrict__ scale,
    const T* __restrict__ res,
    const float* __restrict__ gate,
    T* __restrict__ y,
    float eps,
    float temperature,
    int fan,
    const T* __restrict__ next_w,
    float next_eps,
    int plus_one,
    T* __restrict__ z,
    int rows,
    const u32* __restrict__ win)
{
    constexpr int hidden = 8 * BLOCK;
    const int first = blockIdx.x * ROWS;
    const int end = win != nullptr ? min(rows, static_cast<int>(win[0]) * fan) : rows;
    const int last = min(first + ROWS, end);
    const long long shift = win != nullptr ? static_cast<long long>(win[1]) * fan : 0;
    const int at = threadIdx.x * 8;
    float alpha[8], beta[8];
    if (res != nullptr) {
        float g[8];
        *reinterpret_cast<float4*>(g) = *reinterpret_cast<const float4*>(gate + at);
        *reinterpret_cast<float4*>(g + 4) = *reinterpret_cast<const float4*>(gate + at + 4);
#pragma unroll
        for (int e = 0; e < 8; ++e) norm_gate_ab(g[e], temperature, alpha[e], beta[e]);
    }
    __shared__ float buf[BLOCK];
    uint4 xn{}, rn{};
    if (first < last) {
        xn = *reinterpret_cast<const uint4*>(x + (first + shift) * hidden + at);
        if (res != nullptr) rn = *reinterpret_cast<const uint4*>(res + (first + shift) * hidden + at);
    }
    for (int row = first; row < last; ++row) {
        const long long base = (row + shift) * hidden + at;
        T v[8], rv[8];
        *reinterpret_cast<uint4*>(v) = xn;
        *reinterpret_cast<uint4*>(rv) = rn;
        if (row + 1 < last) {
            xn = *reinterpret_cast<const uint4*>(x + base + hidden);
            if (res != nullptr) rn = *reinterpret_cast<const uint4*>(res + base + hidden);
        }
        float u[8];
        norm_gate_in(v, scale, at, u);
        if (row > first) __syncthreads();
        norm_gate_out<T, BLOCK, NEXT>(
            u, rv, res != nullptr, [&](int e, float& a, float& b) { a = alpha[e], b = beta[e]; },
            base, at, y, eps, next_w, next_eps, plus_one, z, buf);
    }
}

// `rmsnorm_no_scale` a warp a (row, head) unit, landing what the 256-thread
// block form lands bit for bit: lane `l` takes the sums of the block's
// threads `l`, `l + 32`, ... (each its elements `t`, `t + 256`, ... in
// order), folds each as the block's warps fold theirs and the eight as its
// first warp folds them. A 512-wide head (DeepSeek-V4's q) is a warp where
// a block of 256 threads walked it two elements a thread.
template <class T, int RUNS>
__global__ void __launch_bounds__(256) rmsnorm_no_scale_warp(
    const T* __restrict__ x,
    T* __restrict__ y,
    int units,
    float eps,
    int heads,
    const u32* __restrict__ win)
{
    constexpr int hidden = 256 * RUNS;
    const int unit = static_cast<int>(blockIdx.x) * 8 + static_cast<int>(threadIdx.x >> 5);
    if (unit >= units) return;
    if (win != nullptr && unit / heads >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? unit + static_cast<int>(win[1]) * heads : unit;
    const int lane = threadIdx.x & 31;
    const T* xr = x + static_cast<long long>(plane_row) * hidden;
    T* yr = y + static_cast<long long>(plane_row) * hidden;
    float v[8][RUNS];
    float block = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        float local = 0.f;
#pragma unroll
        for (int r = 0; r < RUNS; ++r) {
            v[w][r] = Elem<T>::to_f32(xr[w * 32 + lane + r * 256]);
            local = fmaf(v[w][r], v[w][r], local);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) local += __shfl_xor_sync(0xffffffffu, local, off);
        if (lane == w) block = local;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) block += __shfl_xor_sync(0xffffffffu, block, off);
    const float inv_rms = rsqrtf(block / static_cast<float>(hidden) + eps);
#pragma unroll
    for (int w = 0; w < 8; ++w)
#pragma unroll
        for (int r = 0; r < RUNS; ++r)
            yr[w * 32 + lane + r * 256] = Elem<T>::from_f32(v[w][r] * inv_rms);
}

// `rmsnorm_no_scale` `LANES` lanes a (row, head) unit over 16-byte runs:
// lane `l` loads the head's runs `l`, `l + LANES`, ... (eight elements each),
// so a 512-wide head (DeepSeek-V4's q) is two loads a lane of 32 and a
// 64-wide one (Muse Spark's) one a lane of 8. Its sum folds in a different
// order from the block form's, so it lands within rounding of it.
template <class T, int RUNS, int LANES = 32>
__global__ void __launch_bounds__(256) rmsnorm_no_scale_vec(
    const T* __restrict__ x,
    T* __restrict__ y,
    int units,
    float eps,
    int heads,
    const u32* __restrict__ win)
{
    constexpr int hidden = 8 * LANES * RUNS;
    const int unit = static_cast<int>(blockIdx.x) * (256 / LANES) + static_cast<int>(threadIdx.x) / LANES;
    if (LANES == 32 && unit >= units) return;
    // A unit shares its warp's shuffles with its neighbours, so a dead one
    // runs them on zeros.
    const bool live = unit < units && (win == nullptr || unit / heads < static_cast<int>(win[0]));
    const int plane_row = win != nullptr ? unit + static_cast<int>(win[1]) * heads : unit;
    const int lane = threadIdx.x % LANES;
    const T* xr = x + static_cast<long long>(plane_row) * hidden + lane * 8;
    T* yr = y + static_cast<long long>(plane_row) * hidden + lane * 8;
    T v[RUNS][8];
#pragma unroll
    for (int r = 0; r < RUNS; ++r)
        *reinterpret_cast<uint4*>(v[r]) = live ? *reinterpret_cast<const uint4*>(xr + r * 8 * LANES) : uint4{};
    float local = 0.f;
#pragma unroll
    for (int r = 0; r < RUNS; ++r)
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const float f = Elem<T>::to_f32(v[r][u]);
            local = fmaf(f, f, local);
        }
#pragma unroll
    for (int off = LANES / 2; off > 0; off >>= 1) local += __shfl_xor_sync(0xffffffffu, local, off);
    if (!live) return;
    const float inv_rms = rsqrtf(local / static_cast<float>(hidden) + eps);
#pragma unroll
    for (int r = 0; r < RUNS; ++r) {
        T out[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) out[u] = Elem<T>::from_f32(Elem<T>::to_f32(v[r][u]) * inv_rms);
        *reinterpret_cast<uint4*>(yr + r * 8 * LANES) = *reinterpret_cast<const uint4*>(out);
    }
}

template <class T, int BLOCK = 256>
__global__ void rmsnorm_gated(
    const T* __restrict__ x,
    const T* __restrict__ gate,
    const float* __restrict__ weight,
    T* __restrict__ y,
    int hidden,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const T* xr = x + static_cast<long long>(row) * hidden;
    const T* gr = gate + static_cast<long long>(row) * hidden;
    T* yr = y + static_cast<long long>(row) * hidden;

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float v = Elem<T>::to_f32(xr[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_fast<BLOCK>(local, buf);
    const float inv_rms = rsqrtf(buf_sum / static_cast<float>(hidden) + eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        const float xv = Elem<T>::to_f32(xr[i]) * inv_rms;
        const float wv = weight[i];
        const float gv = Elem<T>::to_f32(gr[i]);

        const float sg = gv / (1.f + __expf(-gv));
        yr[i] = Elem<T>::from_f32(wv * xv * sg);
    }
}

// The gated norm over an f32 accumulator, or one landed in bf16 (`X`).
template <class X, class T, int BLOCK = 256>
__global__ void rmsnorm_gated_in(
    const X* __restrict__ x,
    const T* __restrict__ gate,
    const float* __restrict__ weight,
    T* __restrict__ y,
    int hidden,
    float eps,
    int sigmoid_gate,
    int heads,
    const u32* __restrict__ win)
{
    const int row = blockIdx.x;

    if (win != nullptr && row / heads >= static_cast<int>(win[0])) return;

    const int plane_row =
        win != nullptr ? row + static_cast<int>(win[1]) * heads : row;

    const int tid = threadIdx.x;

    const X* xr = x + static_cast<long long>(plane_row) * hidden;
    const T* gr = gate + static_cast<long long>(plane_row) * hidden;
    T* yr = y + static_cast<long long>(plane_row) * hidden;

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float v = Elem<X>::to_f32(xr[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_fast<BLOCK>(local, buf);
    const float inv_rms = rsqrtf(buf_sum / static_cast<float>(hidden) + eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        const float xv = Elem<X>::to_f32(xr[i]) * inv_rms;
        const float wv = weight != nullptr ? weight[i] : 1.f;
        const float gv = Elem<T>::to_f32(gr[i]);
        const float sg = sigmoid_gate
            ? 1.f / (1.f + __expf(-gv))
            : gv / (1.f + __expf(-gv));
        yr[i] = Elem<T>::from_f32(wv * xv * sg);
    }
}

// The gated norm `LANES` lanes per (row, head) when the normed width is
// `LANES * PER`: a lane carries `PER` consecutive values, so the sum of
// squares is one butterfly and no block barrier; a block is `256 / LANES`
// such rows. No weight: scaleless.
template <class X, class T, int PER, int LANES = 32>
__global__ void __launch_bounds__(256) rmsnorm_gated_in_warp(
    const X* __restrict__ x,
    const T* __restrict__ gate,
    const float* __restrict__ weight,
    T* __restrict__ y,
    int units,
    float eps,
    int sigmoid_gate,
    int heads,
    const u32* __restrict__ win)
{
    static_assert(PER == 4 || PER == 8, "a lane's gate run is one 8- or 16-byte load");
    constexpr int hidden = LANES * PER;
    const int unit = static_cast<int>(blockIdx.x) * (256 / LANES) + static_cast<int>(threadIdx.x) / LANES;
    if (LANES == 32 && unit >= units) return;
    // A unit shares its warp's shuffles with its neighbours: a dead one runs
    // them on zeros, reading and landing nothing.
    const bool live = unit < units && (win == nullptr || unit / heads < static_cast<int>(win[0]));
    const int plane_row = win != nullptr ? unit + static_cast<int>(win[1]) * heads : unit;
    const int lane = threadIdx.x % LANES;
    const long long at = live ? static_cast<long long>(plane_row) * hidden + lane * PER : 0;

    float xv[PER];
    float wv[PER];
    T gv[PER];
    #pragma unroll
    for (int q = 0; q < PER / 4; ++q) {
        if constexpr (sizeof(X) == 4) {
            const float4 xf = reinterpret_cast<const float4*>(x + at)[q];
            xv[4 * q] = xf.x;
            xv[4 * q + 1] = xf.y;
            xv[4 * q + 2] = xf.z;
            xv[4 * q + 3] = xf.w;
        } else {
            const uint2 xb = reinterpret_cast<const uint2*>(x + at)[q];
            const float2 x01 = bf16x2_to_f32(*reinterpret_cast<const bf16x2*>(&xb.x));
            const float2 x23 = bf16x2_to_f32(*reinterpret_cast<const bf16x2*>(&xb.y));
            xv[4 * q] = x01.x;
            xv[4 * q + 1] = x01.y;
            xv[4 * q + 2] = x23.x;
            xv[4 * q + 3] = x23.y;
        }
        const float4 wf = weight != nullptr ? reinterpret_cast<const float4*>(weight + lane * PER)[q]
                                            : make_float4(1.f, 1.f, 1.f, 1.f);
        wv[4 * q] = wf.x;
        wv[4 * q + 1] = wf.y;
        wv[4 * q + 2] = wf.z;
        wv[4 * q + 3] = wf.w;
    }
    if constexpr (PER == 4) {
        *reinterpret_cast<uint2*>(gv) = *reinterpret_cast<const uint2*>(gate + at);
    } else {
        *reinterpret_cast<uint4*>(gv) = *reinterpret_cast<const uint4*>(gate + at);
    }
    float local = 0.f;
    #pragma unroll
    for (int u = 0; u < PER; ++u) local += xv[u] * xv[u];
    #pragma unroll
    for (int off = LANES / 2; off > 0; off >>= 1) local += __shfl_xor_sync(0xffffffffu, local, off);
    if (!live) return;
    const float inv_rms = rsqrtf(local / static_cast<float>(hidden) + eps);

    T out[PER];
    #pragma unroll
    for (int u = 0; u < PER; ++u) {
        const float g = Elem<T>::to_f32(gv[u]);
        // The gate on the fast division (vLLM's Triton gate's).
        const float sg = __fdividef(sigmoid_gate ? 1.f : g, 1.f + __expf(-g));
        out[u] = Elem<T>::from_f32(wv[u] * (xv[u] * inv_rms) * sg);
    }
    if constexpr (PER == 4) {
        *reinterpret_cast<uint2*>(y + at) = *reinterpret_cast<const uint2*>(out);
    } else {
        *reinterpret_cast<uint4*>(y + at) = *reinterpret_cast<const uint4*>(out);
    }
}

template <class T, int BLOCK, int PER_THREAD, bool SCALE, bool POST, bool POST_PLUS_ONE>
__global__ void rmsnorm_residual_add(
    const T* __restrict__ x,
    const T* __restrict__ w0,
    T* __restrict__ t,
    T* __restrict__ y,
    const T* __restrict__ s,
    T* __restrict__ scaled,
    const T* __restrict__ w1,
    T* __restrict__ out,
    int hidden,
    float eps0,
    float eps1,
    const u32* __restrict__ win)
{
    const int row = blockIdx.x;

    if (win != nullptr && row >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? row + static_cast<int>(win[1]) : row;
    const int tid = threadIdx.x;
    const long long base = static_cast<long long>(plane_row) * hidden;

    __shared__ float buf[BLOCK];
    __shared__ float buf2[BLOCK];

    float xv[PER_THREAD];
    float yv[PER_THREAD];
    float local = 0.f;
#pragma unroll
    for (int k = 0; k < PER_THREAD; ++k) {
        const int i = tid + k * BLOCK;
        xv[k] = 0.f;
        yv[k] = 0.f;
        if (i < hidden) {
            xv[k] = Elem<T>::to_f32(x[base + i]);
            yv[k] = Elem<T>::to_f32(y[base + i]);
            local += xv[k] * xv[k];
        }
    }
    const float sum0 = block_reduce_sum_fast<BLOCK>(local, buf);
    const float inv0 = rsqrtf(sum0 / static_cast<float>(hidden) + eps0);

    const float sf = SCALE ? Elem<T>::to_f32(s[0]) : 1.f;

    float last[PER_THREAD];
    float local2 = 0.f;
#pragma unroll
    for (int k = 0; k < PER_THREAD; ++k) {
        const int i = tid + k * BLOCK;
        if (i < hidden) {
            const float wv = Elem<T>::to_f32(w0[i]);
            const T tv = Elem<T>::from_f32(xv[k] * inv0 * wv);
            t[base + i] = tv;
            const T folded = Elem<T>::from_f32(yv[k] + Elem<T>::to_f32(tv));
            y[base + i] = folded;
            T final_v = folded;
            if constexpr (SCALE) {
                final_v = Elem<T>::from_f32(Elem<T>::to_f32(folded) * sf);
                scaled[base + i] = final_v;
            }
            last[k] = Elem<T>::to_f32(final_v);
            local2 += last[k] * last[k];
        } else {
            last[k] = 0.f;
        }
    }
    if constexpr (POST) {
        const float sum1 = block_reduce_sum_fast<BLOCK>(local2, buf2);
        const float inv1 = rsqrtf(sum1 / static_cast<float>(hidden) + eps1);
#pragma unroll
        for (int k = 0; k < PER_THREAD; ++k) {
            const int i = tid + k * BLOCK;
            if (i < hidden) {
                float wv = Elem<T>::to_f32(w1[i]);
                if constexpr (POST_PLUS_ONE) wv += 1.f;
                out[base + i] = Elem<T>::from_f32(last[k] * inv1 * wv);
            }
        }
    }
}

template <int BLOCK, int CHUNKS, bool SCALE, bool POST, bool POST_PLUS_ONE>
__global__ __launch_bounds__(BLOCK) void rmsnorm_residual_add_vec8(
    const bf16* __restrict__ x,
    const bf16* __restrict__ w0,
    bf16* __restrict__ t,
    bf16* __restrict__ y,
    const bf16* __restrict__ s,
    bf16* __restrict__ scaled,
    const bf16* __restrict__ w1,
    bf16* __restrict__ out,
    int hidden,
    float eps0,
    float eps1,
    const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const int row = blockIdx.x;
    if (win != nullptr && row >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? row + static_cast<int>(win[1]) : row;
    const int tid = threadIdx.x;
    const int nvec = hidden / 8;
    const long long base = static_cast<long long>(plane_row) * hidden;

    const uint4* xr = reinterpret_cast<const uint4*>(x + base);
    uint4* yr = reinterpret_cast<uint4*>(y + base);
    uint4* tr = reinterpret_cast<uint4*>(t + base);
    uint4* sr = SCALE ? reinterpret_cast<uint4*>(scaled + base) : nullptr;
    uint4* outr = POST ? reinterpret_cast<uint4*>(out + base) : nullptr;
    const uint4* w0r = reinterpret_cast<const uint4*>(w0);
    const uint4* w1r = POST ? reinterpret_cast<const uint4*>(w1) : nullptr;

    __shared__ float buf[BLOCK / 32 + 1];
    __shared__ float buf2[BLOCK / 32 + 1];

    float xv[CHUNKS][8];
    float yv[CHUNKS][8];
    float local = 0.f;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int i = tid + c * BLOCK;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            xv[c][j] = 0.f;
            yv[c][j] = 0.f;
        }
        if (i < nvec) {
            const uint4 xq = xr[i];
            const uint4 yq = yr[i];
            const bf16x2* xh = reinterpret_cast<const bf16x2*>(&xq);
            const bf16x2* yh = reinterpret_cast<const bf16x2*>(&yq);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 a = bf16x2_to_f32(xh[j]);
                const float2 b = bf16x2_to_f32(yh[j]);
                xv[c][2 * j] = a.x;
                xv[c][2 * j + 1] = a.y;
                yv[c][2 * j] = b.x;
                yv[c][2 * j + 1] = b.y;
                local += a.x * a.x + a.y * a.y;
            }
        }
    }
    const float sum0 = block_reduce_sum_fast<BLOCK>(local, buf);
    const float inv0 = rsqrtf(sum0 / static_cast<float>(hidden) + eps0);
    const float sf = SCALE ? Elem<bf16>::to_f32(s[0]) : 1.f;

    float last[CHUNKS][8];
    float local2 = 0.f;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int i = tid + c * BLOCK;
#pragma unroll
        for (int j = 0; j < 8; ++j) last[c][j] = 0.f;
        if (i < nvec) {
            const uint4 wq = w0r[i];
            const bf16x2* wh = reinterpret_cast<const bf16x2*>(&wq);
            uint4 tq;
            uint4 yq;
            uint4 sq;
            bf16x2* th = reinterpret_cast<bf16x2*>(&tq);
            bf16x2* yh = reinterpret_cast<bf16x2*>(&yq);
            bf16x2* sh = reinterpret_cast<bf16x2*>(&sq);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 w = bf16x2_to_f32(wh[j]);
                th[j] = f32_to_bf16x2(xv[c][2 * j] * inv0 * w.x, xv[c][2 * j + 1] * inv0 * w.y);
                const float2 tv = bf16x2_to_f32(th[j]);
                yh[j] = f32_to_bf16x2(yv[c][2 * j] + tv.x, yv[c][2 * j + 1] + tv.y);
                float2 fin = bf16x2_to_f32(yh[j]);
                if constexpr (SCALE) {
                    sh[j] = f32_to_bf16x2(fin.x * sf, fin.y * sf);
                    fin = bf16x2_to_f32(sh[j]);
                }
                last[c][2 * j] = fin.x;
                last[c][2 * j + 1] = fin.y;
                local2 += fin.x * fin.x + fin.y * fin.y;
            }
            tr[i] = tq;
            yr[i] = yq;
            if constexpr (SCALE) sr[i] = sq;
        }
    }
    if constexpr (POST) {
        const float sum1 = block_reduce_sum_fast<BLOCK>(local2, buf2);
        const float inv1 = rsqrtf(sum1 / static_cast<float>(hidden) + eps1);
#pragma unroll
        for (int c = 0; c < CHUNKS; ++c) {
            const int i = tid + c * BLOCK;
            if (i < nvec) {
                const uint4 wq = w1r[i];
                const bf16x2* wh = reinterpret_cast<const bf16x2*>(&wq);
                uint4 oq;
                bf16x2* oh = reinterpret_cast<bf16x2*>(&oq);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float2 w = bf16x2_to_f32(wh[j]);
                    if constexpr (POST_PLUS_ONE) {
                        w.x += 1.f;
                        w.y += 1.f;
                    }
                    oh[j] = f32_to_bf16x2(last[c][2 * j] * inv1 * w.x, last[c][2 * j + 1] * inv1 * w.y);
                }
                outr[i] = oq;
            }
        }
    }
}

template <class T>
__global__ void residual_add(T* __restrict__ y, const T* __restrict__ x, usize n,
                             int width, const u32* __restrict__ win) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (win != nullptr &&
        i >= static_cast<usize>(win[0]) * static_cast<usize>(width)) return;
    const usize at = win != nullptr
        ? i + static_cast<usize>(win[1]) * static_cast<usize>(width)
        : i;

    const float a = Elem<T>::to_f32(y[at]);
    const float b = Elem<T>::to_f32(x[at]);
    y[at] = Elem<T>::from_f32(a + b);
}

// Eight bf16 lanes a thread, rows of whole 16-byte groups (`width8` a row).
__global__ void residual_add_vec8(bf16* __restrict__ y, const bf16* __restrict__ x, usize n8,
                                  int width8, const u32* __restrict__ win) {
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    if (win != nullptr && i >= static_cast<usize>(win[0]) * static_cast<usize>(width8)) return;
    const usize at = win != nullptr ? i + static_cast<usize>(win[1]) * static_cast<usize>(width8) : i;
    float4 a = reinterpret_cast<const float4*>(y)[at];
    const float4 b = reinterpret_cast<const float4*>(x)[at];
    bf16x2* ah = reinterpret_cast<bf16x2*>(&a);
    const bf16x2* bh = reinterpret_cast<const bf16x2*>(&b);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 fa = bf16x2_to_f32(ah[j]);
        const float2 fb = bf16x2_to_f32(bh[j]);
        ah[j] = f32_to_bf16x2(fa.x + fb.x, fa.y + fb.y);
    }
    reinterpret_cast<float4*>(y)[at] = a;
}

template <class T>
__global__ void mul_scalar(T* __restrict__ x, float s, usize n,
                           int width, const u32* __restrict__ win) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (win != nullptr &&
        i >= static_cast<usize>(win[0]) * static_cast<usize>(width)) return;
    const usize at = win != nullptr
        ? i + static_cast<usize>(win[1]) * static_cast<usize>(width)
        : i;

    const float s_rounded = Elem<T>::to_f32(Elem<T>::from_f32(s));
    x[at] = Elem<T>::from_f32(Elem<T>::to_f32(x[at]) * s_rounded);
}

template <class T>
__global__ void silu_scaled(T* __restrict__ x, float s, usize n,
                            int width, const u32* __restrict__ win) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (win != nullptr &&
        i >= static_cast<usize>(win[0]) * static_cast<usize>(width)) return;
    const usize at = win != nullptr
        ? i + static_cast<usize>(win[1]) * static_cast<usize>(width)
        : i;

    const float v = Elem<T>::to_f32(x[at]) * s;
    x[at] = Elem<T>::from_f32(v / (1.f + __expf(-v)));
}

template <class T>
__global__ void scale(T* __restrict__ x, const T* __restrict__ s, usize n,
                      int width, const u32* __restrict__ win) {
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (win != nullptr &&
        i >= static_cast<usize>(win[0]) * static_cast<usize>(width)) return;
    const usize at = win != nullptr
        ? i + static_cast<usize>(win[1]) * static_cast<usize>(width)
        : i;

    const float f = Elem<T>::to_f32(s[0]);
    x[at] = Elem<T>::from_f32(Elem<T>::to_f32(x[at]) * f);
}

template <class T, class TB = T>
__device__ __forceinline__ void add_bias_row(
    T* __restrict__ row,
    const TB* __restrict__ bias,
    int dim)
{
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        const float v = Elem<T>::to_f32(row[d]) + Elem<TB>::to_f32(bias[d]);
        row[d] = Elem<T>::from_f32(v);
    }
}

template <class T, class TB = T>
__global__ void add_bias(
    T* __restrict__ out,
    const TB* __restrict__ bias,
    int dim,
    const u32* __restrict__ win)
{
    const int n = static_cast<int>(blockIdx.x);

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    add_bias_row<T, TB>(out + static_cast<long long>(row) * dim, bias, dim);
}

template <class T>
__global__ void standardize(
    T* __restrict__ out,
    const T* __restrict__ bias,
    const T* __restrict__ scale,
    int dim)
{
    T* __restrict__ row = out + static_cast<long long>(blockIdx.x) * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {

        const float v = (Elem<T>::to_f32(row[d]) - Elem<T>::to_f32(bias[d]))
                      * Elem<T>::to_f32(scale[d]);
        row[d] = Elem<T>::from_f32(v);
    }
}

template <class T>
__global__ void add_bias_strided(
    T* __restrict__ out,
    const T* __restrict__ bias,
    int dim,
    int stride)
{
    add_bias_row<T>(out + static_cast<long long>(blockIdx.x) * stride, bias, dim);
}

constexpr int kMaxBlocks = 32;

constexpr int kThreads = 256;

__device__ __forceinline__ float block_reduce_sum(float x, float* scratch) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffffu, x, offset);
    }
    const int lane = threadIdx.x & (warpSize - 1);
    const int warp = threadIdx.x / warpSize;
    if (lane == 0) scratch[warp] = x;
    __syncthreads();
    const int warps = blockDim.x / warpSize;
    float total = 0.f;
    if (threadIdx.x == 0) {
        for (int w = 0; w < warps; ++w) total += scratch[w];
        scratch[0] = total;
    }
    __syncthreads();
    total = scratch[0];
    __syncthreads();
    return total;
}

// The closed blocks a blend reads, each its own plane (`rows x H`), by value.
struct ResBlocks {
    const void* at[kMaxBlocks];
};

// A row's eight elements from 16 bytes, or its one element where the row is
// not read 16 bytes at a time.
template <class T>
__device__ __forceinline__ void res_load8(const T* v, i32 h, float (&x)[8]) {
    const uint4 w = *reinterpret_cast<const uint4*>(v + h);
    const T* e = reinterpret_cast<const T*>(&w);
#pragma unroll
    for (int i = 0; i < 8; ++i) x[i] = Elem<T>::to_f32(e[i]);
}

// AttnRes's blend of a token's closed blocks and its running prefix: each
// candidate's RMS-normed dot with the projection (one warp a candidate, its
// square sum and its dot in one pass), their softmax, and the blend of the
// candidates under it.
// With `kNorm` the blended row is RMS-normed under `out_weight` before it
// lands (`rmsnorm` of the blend's bf16 row, the two in one pass): its
// columns held in registers, at most `kTh * 8 * kChunks` of them, 16-byte
// runs only.
template <class T, int kTh, bool kNorm>
__device__ __forceinline__ void res_blend_body(
    const T* __restrict__ prefix,
    const ResBlocks& blocks,
    const T* __restrict__ norm_weight,
    const T* __restrict__ proj_weight,
    T* __restrict__ out,
    i32 B, i32 H, float eps,
    const T* __restrict__ out_weight, float out_eps,
    const u32* __restrict__ win)
{
    const i32 t = static_cast<i32>(blockIdx.x);

    if (win != nullptr && t >= static_cast<i32>(win[0])) return;

    const i32 row = win != nullptr ? t + static_cast<i32>(win[1]) : t;

    __shared__ float prob_s[kMaxBlocks + 1];

    const long long token_off = static_cast<long long>(row) * H;
    const i32 rows = B + 1;

    auto row_ptr = [&](i32 j) -> const T* {
        return (j < B) ? static_cast<const T*>(blocks.at[j]) + token_off : prefix + token_off;
    };
    // 16-byte runs where every row and both weights allow them.
    bool wide = (H % 8) == 0
        && (reinterpret_cast<unsigned long long>(norm_weight) & 15ull) == 0
        && (reinterpret_cast<unsigned long long>(proj_weight) & 15ull) == 0
        && (reinterpret_cast<unsigned long long>(out + token_off) & 15ull) == 0;
    for (i32 j = 0; j < rows; ++j) {
        wide = wide && (reinterpret_cast<unsigned long long>(row_ptr(j)) & 15ull) == 0;
    }

    // Each thread walks its columns of a group of candidates at once (their
    // loads independent of one another), its square sums and dots reduced
    // across the block a group at a time.
    constexpr int kGroup = 8;
    __shared__ float part[kTh / 32][2 * kGroup];
    const i32 lane = static_cast<i32>(threadIdx.x) & 31;
    const i32 warp = static_cast<i32>(threadIdx.x) >> 5;
    const i32 warps = static_cast<i32>(blockDim.x) >> 5;
    for (i32 j0 = 0; j0 < rows; j0 += kGroup) {
        float ss[kGroup];
        float dot[kGroup];
#pragma unroll
        for (int g = 0; g < kGroup; ++g) {
            ss[g] = 0.f;
            dot[g] = 0.f;
        }
        if (wide) {
            for (i32 h = static_cast<i32>(threadIdx.x) * 8; h < H;
                 h += static_cast<i32>(blockDim.x) * 8) {
                float nw[8], pw[8];
                res_load8(norm_weight, h, nw);
                res_load8(proj_weight, h, pw);
#pragma unroll
                for (int g = 0; g < kGroup; ++g) {
                    if (j0 + g < rows) {
                        float x[8];
                        res_load8(row_ptr(j0 + g), h, x);
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            ss[g] += x[i] * x[i];
                            dot[g] += x[i] * nw[i] * pw[i];
                        }
                    }
                }
            }
        } else {
            for (i32 h = static_cast<i32>(threadIdx.x); h < H;
                 h += static_cast<i32>(blockDim.x)) {
                const float w = Elem<T>::to_f32(norm_weight[h]) * Elem<T>::to_f32(proj_weight[h]);
#pragma unroll
                for (int g = 0; g < kGroup; ++g) {
                    if (j0 + g < rows) {
                        const float x = Elem<T>::to_f32(row_ptr(j0 + g)[h]);
                        ss[g] += x * x;
                        dot[g] += x * w;
                    }
                }
            }
        }
#pragma unroll
        for (int g = 0; g < kGroup; ++g) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                ss[g] += __shfl_xor_sync(0xffffffffu, ss[g], off);
                dot[g] += __shfl_xor_sync(0xffffffffu, dot[g], off);
            }
        }
        if (lane == 0) {
#pragma unroll
            for (int g = 0; g < kGroup; ++g) {
                part[warp][g] = ss[g];
                part[warp][kGroup + g] = dot[g];
            }
        }
        __syncthreads();
        if (threadIdx.x < kGroup && j0 + static_cast<i32>(threadIdx.x) < rows) {
            float s2 = 0.f;
            float d = 0.f;
            for (i32 w = 0; w < warps; ++w) {
                s2 += part[w][threadIdx.x];
                d += part[w][kGroup + threadIdx.x];
            }
            prob_s[j0 + threadIdx.x] = d * rsqrtf(s2 / static_cast<float>(H) + eps);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float m = prob_s[0];
        for (i32 j = 1; j < rows; ++j) m = fmaxf(m, prob_s[j]);
        float sum = 0.f;
        for (i32 j = 0; j < rows; ++j) {
            prob_s[j] = __expf(prob_s[j] - m);
            sum += prob_s[j];
        }
        const float inv = 1.f / sum;
        for (i32 j = 0; j < rows; ++j) prob_s[j] *= inv;
    }
    __syncthreads();

    if constexpr (kNorm) {
        // The blended row in registers, rounded to bf16 as it would land,
        // its square sum, then the norm.
        constexpr int kChunks = 4;
        float hv[kChunks][8];
        float ss = 0.f;
#pragma unroll
        for (int c = 0; c < kChunks; ++c) {
            const i32 h = (static_cast<i32>(threadIdx.x) + c * kTh) * 8;
#pragma unroll
            for (int i = 0; i < 8; ++i) hv[c][i] = 0.f;
            if (h < H) {
                for (i32 j = 0; j < rows; ++j) {
                    float x[8];
                    res_load8(row_ptr(j), h, x);
                    const float p = prob_s[j];
#pragma unroll
                    for (int i = 0; i < 8; ++i) hv[c][i] += p * x[i];
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    hv[c][i] = Elem<T>::to_f32(Elem<T>::from_f32(hv[c][i]));
                    ss += hv[c][i] * hv[c][i];
                }
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
        if (lane == 0) part[warp][0] = ss;
        __syncthreads();
        float total = 0.f;
        for (i32 w = 0; w < warps; ++w) total += part[w][0];
        const float inv_rms = rsqrtf(total / static_cast<float>(H) + out_eps);
#pragma unroll
        for (int c = 0; c < kChunks; ++c) {
            const i32 h = (static_cast<i32>(threadIdx.x) + c * kTh) * 8;
            if (h < H) {
                float wv[8];
                res_load8(out_weight, h, wv);
                uint4 o;
                T* e = reinterpret_cast<T*>(&o);
#pragma unroll
                for (int i = 0; i < 8; ++i) e[i] = Elem<T>::from_f32(hv[c][i] * inv_rms * wv[i]);
                *reinterpret_cast<uint4*>(out + token_off + h) = o;
            }
        }
        return;
    }
    if (wide) {
        for (i32 h = static_cast<i32>(threadIdx.x) * 8; h < H;
             h += static_cast<i32>(blockDim.x) * 8) {
            float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            for (i32 j = 0; j < rows; ++j) {
                float x[8];
                res_load8(row_ptr(j), h, x);
                const float p = prob_s[j];
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[i] += p * x[i];
            }
            uint4 w;
            T* e = reinterpret_cast<T*>(&w);
#pragma unroll
            for (int i = 0; i < 8; ++i) e[i] = Elem<T>::from_f32(acc[i]);
            *reinterpret_cast<uint4*>(out + token_off + h) = w;
        }
    } else {
        for (i32 h = static_cast<i32>(threadIdx.x); h < H;
             h += static_cast<i32>(blockDim.x)) {
            float acc = 0.f;
            for (i32 j = 0; j < rows; ++j) {
                acc += prob_s[j] * Elem<T>::to_f32(row_ptr(j)[h]);
            }
            out[token_off + h] = Elem<T>::from_f32(acc);
        }
    }
}

template <class T>
__global__ void res_blend(
    const T* __restrict__ prefix,
    ResBlocks blocks,
    const T* __restrict__ norm_weight,
    const T* __restrict__ proj_weight,
    T* __restrict__ out,
    i32 B, i32 H, i32 block_rows, float eps,
    const u32* __restrict__ win)
{
    (void)block_rows;
    res_blend_body<T, kThreads, false>(
        prefix, blocks, norm_weight, proj_weight, out, B, H, eps, nullptr, 0.f, win);
}

// `res_blend` and the `rmsnorm` of its row (under `out_weight`, `out_eps`)
// in one pass, `kTh` threads a row; `out` the normed row.
template <class T, int kTh>
__global__ void __launch_bounds__(kTh) res_blend_norm(
    const T* __restrict__ prefix,
    ResBlocks blocks,
    const T* __restrict__ norm_weight,
    const T* __restrict__ proj_weight,
    const T* __restrict__ out_weight,
    T* __restrict__ out,
    i32 B, i32 H, float eps, float out_eps,
    const u32* __restrict__ win)
{
    res_blend_body<T, kTh, true>(
        prefix, blocks, norm_weight, proj_weight, out, B, H, eps, out_weight, out_eps, win);
}

// A warp `kP` (row, head) pairs: a pair's `D` (32 to 256) dims a lane
// `D / 32` of them, held across the norm. At 128 dims a warp's pairs load
// at once.
template <class ElemT, int kP>
__global__ void __launch_bounds__(256) rmsnorm_gated_by(
    const float* __restrict__ o,
    const ElemT* __restrict__ g,
    const float* __restrict__ weight,
    ElemT* __restrict__ out,
    int rows, int H, int D, float eps,
    const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const long long first = (static_cast<long long>(blockIdx.x) * 8 + (threadIdx.x >> 5)) * kP;
    const int lane = threadIdx.x & 31;
    const int live = win != nullptr ? min(static_cast<int>(win[0]), rows) : rows;
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;
    auto base_of = [&](long long pair, bool& in) {
        const int t = static_cast<int>(pair / H);
        in = t < live;
        return (static_cast<long long>(t + row0) * H + pair % H) * D;
    };
    auto sigmoid = [](float x) { return 1.f / (1.f + __expf(-x)); };
    const int per = D / 32;
    if (per == 4) {
        float4 v[kP];
        uint2 gw[kP];
        long long base[kP];
        bool in[kP];
        #pragma unroll
        for (int p = 0; p < kP; ++p) {
            base[p] = base_of(first + p, in[p]);
            if (in[p]) {
                v[p] = reinterpret_cast<const float4*>(o + base[p])[lane];
                gw[p] = reinterpret_cast<const uint2*>(g + base[p])[lane];
            }
        }
        const float4 wv = reinterpret_cast<const float4*>(weight)[lane];
        const float w4[4] = {wv.x, wv.y, wv.z, wv.w};
        #pragma unroll
        for (int p = 0; p < kP; ++p) {
            if (!in[p]) continue;
            const float x[4] = {v[p].x, v[p].y, v[p].z, v[p].w};
            float acc = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
            const float scale = rsqrtf(acc / static_cast<float>(D) + eps);
            const ElemT* g4 = reinterpret_cast<const ElemT*>(&gw[p]);
            uint2 ow;
            ElemT* o4 = reinterpret_cast<ElemT*>(&ow);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                o4[i] = Elem<ElemT>::from_f32(x[i] * scale * w4[i] * sigmoid(Elem<ElemT>::to_f32(g4[i])));
            reinterpret_cast<uint2*>(out + base[p])[lane] = ow;
        }
        return;
    }
    for (int p = 0; p < kP; ++p) {
        bool in;
        const long long base = base_of(first + p, in);
        if (!in) return;
        float x[8];
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (i < per) x[i] = o[base + lane * per + i];
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (i < per) acc += x[i] * x[i];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
        const float scale = rsqrtf(acc / static_cast<float>(D) + eps);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (i < per) {
                const int d = lane * per + i;
                out[base + d] = Elem<ElemT>::from_f32(x[i] * scale * weight[d] * sigmoid(Elem<ElemT>::to_f32(g[base + d])));
            }
        }
    }
}
}
