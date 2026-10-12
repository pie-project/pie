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

template <class T, int BLOCK = 256>
__global__ void rmsnorm_grouped_plus_one(
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
        const float wv = Elem<T>::to_f32(wr[i]) + 1.f;
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

template <class T, int BLOCK = 256>
__global__ void rmsnorm_gated_f32_in(
    const float* __restrict__ x,
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

    const float* xr = x + static_cast<long long>(plane_row) * hidden;
    const T* gr = gate + static_cast<long long>(plane_row) * hidden;
    T* yr = y + static_cast<long long>(plane_row) * hidden;

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float v = xr[i];
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_fast<BLOCK>(local, buf);
    const float inv_rms = rsqrtf(buf_sum / static_cast<float>(hidden) + eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        const float xv = xr[i] * inv_rms;
        const float wv = weight[i];
        const float gv = Elem<T>::to_f32(gr[i]);
        const float sg = sigmoid_gate
            ? 1.f / (1.f + __expf(-gv))
            : gv / (1.f + __expf(-gv));
        yr[i] = Elem<T>::from_f32(wv * xv * sg);
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

template <class T>
__global__ void res_blend(
    const T* __restrict__ prefix,
    const T* __restrict__ blocks,
    const T* __restrict__ norm_weight,
    const T* __restrict__ proj_weight,
    T* __restrict__ out,
    i32 B, i32 H, i32 block_rows, float eps,
    const u32* __restrict__ win)
{
    const i32 t = static_cast<i32>(blockIdx.x);

    if (win != nullptr && t >= static_cast<i32>(win[0])) return;

    const i32 row = win != nullptr ? t + static_cast<i32>(win[1]) : t;

    __shared__ float scratch[kThreads / 32];
    __shared__ float prob_s[kMaxBlocks + 1];

    const long long token_off = static_cast<long long>(row) * H;
    const i32 rows = B + 1;

    auto row_ptr = [&](i32 j) -> const T* {
        return (j < B) ? blocks + (static_cast<long long>(j) * block_rows + row) * H
                       : prefix + token_off;
    };

    for (i32 j = 0; j < rows; ++j) {
        const T* v = row_ptr(j);
        float ss = 0.f;
        for (i32 h = static_cast<i32>(threadIdx.x); h < H;
             h += static_cast<i32>(blockDim.x)) {
            const float x = Elem<T>::to_f32(v[h]);
            ss += x * x;
        }
        ss = block_reduce_sum(ss, scratch);
        const float scale = rsqrtf(ss / static_cast<float>(H) + eps);

        float dot = 0.f;
        for (i32 h = static_cast<i32>(threadIdx.x); h < H;
             h += static_cast<i32>(blockDim.x)) {
            dot += Elem<T>::to_f32(v[h]) * scale *
                   Elem<T>::to_f32(norm_weight[h]) *
                   Elem<T>::to_f32(proj_weight[h]);
        }
        dot = block_reduce_sum(dot, scratch);
        if (threadIdx.x == 0) {
            prob_s[j] = dot;
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

    for (i32 h = static_cast<i32>(threadIdx.x); h < H;
         h += static_cast<i32>(blockDim.x)) {
        float acc = 0.f;
        for (i32 j = 0; j < rows; ++j) {
            acc += prob_s[j] * Elem<T>::to_f32(row_ptr(j)[h]);
        }
        out[token_off + h] = Elem<T>::from_f32(acc);
    }
}


template <class ElemT>
__global__ void rmsnorm_gated_by(
    const float* __restrict__ o,
    const ElemT* __restrict__ g,
    const float* __restrict__ weight,
    ElemT* __restrict__ out,
    int H, int D, float eps,
    const u32* __restrict__ win)
{
    const int t = blockIdx.x;

    if (win != nullptr && t >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? t + static_cast<int>(win[1]) : t;

    const int h = blockIdx.y;
    const long long base = ((long long)row * H + h) * D;

    float acc = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float x = o[base + d];
        acc += x * x;
    }

    __shared__ float warp_sums[32];
    __shared__ float ssum;

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if ((threadIdx.x & (warpSize - 1)) == 0) warp_sums[threadIdx.x / warpSize] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.f;
        const int warps = (blockDim.x + warpSize - 1) / warpSize;
        for (int w = 0; w < warps; ++w) total += warp_sums[w];
        ssum = total;
    }
    __syncthreads();

    const float scale = rsqrtf(ssum / static_cast<float>(D) + eps);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float gate = Elem<ElemT>::to_f32(g[base + d]);
        const float y = o[base + d] * scale * weight[d] * (1.f / (1.f + __expf(-gate)));
        out[base + d] = Elem<ElemT>::from_f32(y);
    }
}
}
