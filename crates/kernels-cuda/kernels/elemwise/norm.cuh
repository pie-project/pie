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

template <class T, int BLOCK, bool WEIGHT_PLUS_ONE>
__device__ __forceinline__ void rmsnorm_row(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int hidden,
    int x_row_stride,
    int y_row_stride,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const T* xr = x + static_cast<long long>(row) * x_row_stride;
    T* yr = y + static_cast<long long>(row) * y_row_stride;

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float v = Elem<T>::to_f32(xr[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);

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
    float eps)
{
    rmsnorm_row<T, BLOCK, false>(
        x, weight, y, hidden, x_row_stride, y_row_stride, eps);
}

template <class T, int BLOCK = 256>
__global__ void rmsnorm_plus_one(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int hidden,
    int x_row_stride,
    int y_row_stride,
    float eps)
{
    rmsnorm_row<T, BLOCK, true>(
        x, weight, y, hidden, x_row_stride, y_row_stride, eps);
}

// The hyper-connection norm: moments per `group`-wide slice, scale by
// `weight + 1` over the row's FULL width — the weight is indexed by the
// slice, where the per-head norms share one plane across every head. One
// block per (row, group), laid out as `rows x groups` consecutive blocks,
// which is the same flattening `rows_per_head` launches.
template <class T, int BLOCK = 256>
__global__ void rmsnorm_grouped_plus_one(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int group,
    int groups,
    float eps)
{
    const int b = blockIdx.x;
    const int tid = threadIdx.x;

    const T* xr = x + static_cast<long long>(b) * group;
    const T* wr = weight + static_cast<long long>(b % groups) * group;
    T* yr = y + static_cast<long long>(b) * group;

    float local = 0.f;
    for (int i = tid; i < group; i += BLOCK) {
        const float v = Elem<T>::to_f32(xr[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);
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
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);
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

template <int BLOCK>
__global__ void residual_add_rmsnorm_vec8(
    bf16* __restrict__ hidden,
    const bf16* __restrict__ residual,
    const bf16* __restrict__ weight,
    bf16* __restrict__ norm_out,
    int hidden_size,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nvec = hidden_size / 8;
    const long long base = static_cast<long long>(row) * hidden_size;

    float4* hr = reinterpret_cast<float4*>(hidden + base);
    const float4* rr = reinterpret_cast<const float4*>(residual + base);
    float4* nr = reinterpret_cast<float4*>(norm_out + base);
    const float4* wr = reinterpret_cast<const float4*>(weight);

    float local = 0.f;
    for (int i = tid; i < nvec; i += BLOCK) {
        float4 hv = hr[i];
        float4 rv = rr[i];
        bf16x2* hh = reinterpret_cast<bf16x2*>(&hv);
        const bf16x2* rh = reinterpret_cast<const bf16x2*>(&rv);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 a = bf16x2_to_f32(hh[j]);
            const float2 b = bf16x2_to_f32(rh[j]);
            hh[j] = f32_to_bf16x2(a.x + b.x, a.y + b.y);
            const float2 f = bf16x2_to_f32(hh[j]);
            local += f.x * f.x + f.y * f.y;
        }
        hr[i] = hv;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);
    const float inv_rms =
        rsqrtf(buf_sum / static_cast<float>(hidden_size) + eps);

    for (int i = tid; i < nvec; i += BLOCK) {
        float4 v = hr[i];
        float4 g = wr[i];
        float4 o;
        const bf16x2* hv = reinterpret_cast<const bf16x2*>(&v);
        const bf16x2* hg = reinterpret_cast<const bf16x2*>(&g);
        bf16x2* ho = reinterpret_cast<bf16x2*>(&o);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 a = bf16x2_to_f32(hv[j]);
            const float2 b = bf16x2_to_f32(hg[j]);
            ho[j] = f32_to_bf16x2(a.x * inv_rms * b.x,
                                  a.y * inv_rms * b.y);
        }
        nr[i] = o;
    }
}

template <class T, int BLOCK = 256>
__global__ void residual_add_rmsnorm(
    T* __restrict__ hidden,
    const T* __restrict__ residual,
    const T* __restrict__ weight,
    T* __restrict__ norm_out,
    int hidden_size,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    T* hr = hidden + static_cast<long long>(row) * hidden_size;
    const T* rr = residual + static_cast<long long>(row) * hidden_size;
    T* nr = norm_out + static_cast<long long>(row) * hidden_size;

    float local = 0.f;
    for (int i = tid; i < hidden_size; i += BLOCK) {
        const float sum = Elem<T>::to_f32(hr[i]) + Elem<T>::to_f32(rr[i]);
        const T rounded = Elem<T>::from_f32(sum);
        hr[i] = rounded;
        const float v = Elem<T>::to_f32(rounded);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);

    const float inv_rms =
        rsqrtf(buf_sum / static_cast<float>(hidden_size) + eps);

    for (int i = tid; i < hidden_size; i += BLOCK) {
        const float xv = Elem<T>::to_f32(hr[i]);
        const float wv = Elem<T>::to_f32(weight[i]);
        nr[i] = Elem<T>::from_f32(xv * inv_rms * wv);
    }
}

template <class T, int BLOCK = 256>
__global__ void residual_add_scale_rmsnorm(
    T* __restrict__ hidden,
    const T* __restrict__ residual,
    float scale,
    const T* __restrict__ weight,
    T* __restrict__ norm_out,
    int hidden_size,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    T* hr = hidden + static_cast<long long>(row) * hidden_size;
    const T* rr = residual + static_cast<long long>(row) * hidden_size;
    T* nr = norm_out + static_cast<long long>(row) * hidden_size;
    const float scale_rounded = Elem<T>::to_f32(Elem<T>::from_f32(scale));

    float local = 0.f;
    for (int i = tid; i < hidden_size; i += BLOCK) {
        const float sum = Elem<T>::to_f32(hr[i]) + Elem<T>::to_f32(rr[i]);
        const T rounded_sum = Elem<T>::from_f32(sum);
        const T scaled =
            Elem<T>::from_f32(Elem<T>::to_f32(rounded_sum) * scale_rounded);
        hr[i] = scaled;
        const float v = Elem<T>::to_f32(scaled);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);

    const float inv_rms =
        rsqrtf(buf_sum / static_cast<float>(hidden_size) + eps);

    for (int i = tid; i < hidden_size; i += BLOCK) {
        const float xv = Elem<T>::to_f32(hr[i]);
        const float wv = Elem<T>::to_f32(weight[i]);
        nr[i] = Elem<T>::from_f32(xv * inv_rms * wv);
    }
}

template <class T, int BLOCK = 256>
__global__ void rmsnorm_residual_add(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ hidden,
    int hidden_size,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const T* xr = x + static_cast<long long>(row) * hidden_size;
    T* hr = hidden + static_cast<long long>(row) * hidden_size;

    float local = 0.f;
    for (int i = tid; i < hidden_size; i += BLOCK) {
        const float v = Elem<T>::to_f32(xr[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);

    const float inv_rms =
        rsqrtf(buf_sum / static_cast<float>(hidden_size) + eps);
    for (int i = tid; i < hidden_size; i += BLOCK) {
        const T norm = Elem<T>::from_f32(
            Elem<T>::to_f32(xr[i]) * inv_rms * Elem<T>::to_f32(weight[i]));
        hr[i] = Elem<T>::from_f32(
            Elem<T>::to_f32(hr[i]) + Elem<T>::to_f32(norm));
    }
}

template <int BLOCK>
__global__ void rmsnorm_rasr_vec8(
    const bf16* __restrict__ x,
    const bf16* __restrict__ weight,
    bf16* __restrict__ hidden,
    float scale,
    const bf16* __restrict__ next_weight,
    bf16* __restrict__ norm_out,
    int hidden_size,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int vecs = hidden_size / 8;
    const float4* xr = reinterpret_cast<const float4*>(x + (long long)row * hidden_size);
    const float4* wv = reinterpret_cast<const float4*>(weight);
    const float4* nwv = reinterpret_cast<const float4*>(next_weight);
    float4* hr = reinterpret_cast<float4*>(hidden + (long long)row * hidden_size);
    float4* nr = reinterpret_cast<float4*>(norm_out + (long long)row * hidden_size);

    float local = 0.f;
    for (int i = tid; i < vecs; i += BLOCK) {
        const float4 v = xr[i];
        const bf16* b = reinterpret_cast<const bf16*>(&v);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float f = bf16_to_f32(b[j]);
            local += f * f;
        }
    }
    __shared__ float buf[BLOCK];
    const float s0 = block_reduce_sum_exact<BLOCK>(local, buf);
    const float inv_rms = rsqrtf(s0 / static_cast<float>(hidden_size) + eps);
    const float scale_rounded = bf16_to_f32(f32_to_bf16(scale));

    float local_next = 0.f;
    for (int i = tid; i < vecs; i += BLOCK) {
        const float4 xv4 = xr[i];
        const float4 wv4 = wv[i];
        float4 hv4 = hr[i];
        const bf16* xb = reinterpret_cast<const bf16*>(&xv4);
        const bf16* wb = reinterpret_cast<const bf16*>(&wv4);
        bf16* hb = reinterpret_cast<bf16*>(&hv4);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const bf16 norm = f32_to_bf16(
                bf16_to_f32(xb[j]) * inv_rms * bf16_to_f32(wb[j]));
            const float sum = bf16_to_f32(hb[j]) + bf16_to_f32(norm);
            const bf16 rounded = f32_to_bf16(sum);
            const bf16 scaled = f32_to_bf16(bf16_to_f32(rounded) * scale_rounded);
            hb[j] = scaled;
            const float f = bf16_to_f32(scaled);
            local_next += f * f;
        }
        hr[i] = hv4;
    }
    __shared__ float buf2[BLOCK];
    const float s1 = block_reduce_sum_exact<BLOCK>(local_next, buf2);
    const float inv_next = rsqrtf(s1 / static_cast<float>(hidden_size) + eps);

    for (int i = tid; i < vecs; i += BLOCK) {
        const float4 hv4 = hr[i];
        const float4 nw4 = nwv[i];
        const bf16* hb = reinterpret_cast<const bf16*>(&hv4);
        const bf16* nb = reinterpret_cast<const bf16*>(&nw4);
        float4 out4;
        bf16* ob = reinterpret_cast<bf16*>(&out4);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            ob[j] = f32_to_bf16(bf16_to_f32(hb[j]) * inv_next * bf16_to_f32(nb[j]));
        }
        nr[i] = out4;
    }
}

template <class T, int BLOCK>
__global__ void rmsnorm_residual_add_scale_rmsnorm(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ hidden,
    float scale,
    const T* __restrict__ next_weight,
    T* __restrict__ norm_out,
    int hidden_size,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const T* xr = x + static_cast<long long>(row) * hidden_size;
    T* hr = hidden + static_cast<long long>(row) * hidden_size;
    T* nr = norm_out + static_cast<long long>(row) * hidden_size;

    float local = 0.f;
    for (int i = tid; i < hidden_size; i += BLOCK) {
        const float v = Elem<T>::to_f32(xr[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);

    const float inv_rms =
        rsqrtf(buf_sum / static_cast<float>(hidden_size) + eps);
    const float scale_rounded = Elem<T>::to_f32(Elem<T>::from_f32(scale));
    float local_next = 0.f;
    for (int i = tid; i < hidden_size; i += BLOCK) {
        const T norm = Elem<T>::from_f32(
            Elem<T>::to_f32(xr[i]) * inv_rms * Elem<T>::to_f32(weight[i]));
        const float sum = Elem<T>::to_f32(hr[i]) + Elem<T>::to_f32(norm);
        const T rounded_sum = Elem<T>::from_f32(sum);
        const T scaled =
            Elem<T>::from_f32(Elem<T>::to_f32(rounded_sum) * scale_rounded);
        hr[i] = scaled;
        const float v = Elem<T>::to_f32(scaled);
        local_next += v * v;
    }

    __shared__ float buf_next[BLOCK];
    const float buf_next_sum = block_reduce_sum_exact<BLOCK>(local_next, buf_next);

    const float inv_next =
        rsqrtf(buf_next_sum / static_cast<float>(hidden_size) + eps);
    for (int i = tid; i < hidden_size; i += BLOCK) {
        nr[i] = Elem<T>::from_f32(
            Elem<T>::to_f32(hr[i]) * inv_next * Elem<T>::to_f32(next_weight[i]));
    }
}

template <class T, int BLOCK = 256>
__global__ void rmsnorm_no_scale(
    const T* __restrict__ x,
    T* __restrict__ y,
    int hidden,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const T* xr = x + static_cast<long long>(row) * hidden;
    T* yr = y + static_cast<long long>(row) * hidden;

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float v = Elem<T>::to_f32(xr[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);
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
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);
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
    int sigmoid_gate)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float* xr = x + static_cast<long long>(row) * hidden;
    const T* gr = gate + static_cast<long long>(row) * hidden;
    T* yr = y + static_cast<long long>(row) * hidden;

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float v = xr[i];
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);
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

template <class T>
__global__ void residual_add(T* __restrict__ y, const T* __restrict__ x, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = Elem<T>::to_f32(y[i]);
    const float b = Elem<T>::to_f32(x[i]);
    y[i] = Elem<T>::from_f32(a + b);
}

template <class T>
__global__ void mul_scalar(T* __restrict__ x, float s, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float s_rounded = Elem<T>::to_f32(Elem<T>::from_f32(s));
    x[i] = Elem<T>::from_f32(Elem<T>::to_f32(x[i]) * s_rounded);
}

// silu(s * x), in place: the scalar sits INSIDE the activation, which is
// what keeps this from being `mul_scalar` composed with anything.
template <class T>
__global__ void silu_scaled(T* __restrict__ x, float s, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = Elem<T>::to_f32(x[i]) * s;
    x[i] = Elem<T>::from_f32(v / (1.f + __expf(-v)));
}

template <class T>
__global__ void scale(T* __restrict__ x, const T* __restrict__ s, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float f = Elem<T>::to_f32(s[0]);
    x[i] = Elem<T>::from_f32(Elem<T>::to_f32(x[i]) * f);
}

template <class T>
__device__ __forceinline__ void add_bias_row(
    T* __restrict__ row,
    const T* __restrict__ bias,
    int dim)
{
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        const float v = Elem<T>::to_f32(row[d]) + Elem<T>::to_f32(bias[d]);
        row[d] = Elem<T>::from_f32(v);
    }
}

template <class T>
__global__ void add_bias(
    T* __restrict__ out,
    const T* __restrict__ bias,
    int dim)
{
    add_bias_row<T>(out + static_cast<long long>(blockIdx.x) * dim, bias, dim);
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
    i32 B, i32 H, i32 block_rows, float eps)
{
    const i32 t = static_cast<i32>(blockIdx.x);

    __shared__ float scratch[kThreads / 32];
    __shared__ float prob_s[kMaxBlocks + 1];

    const long long token_off = static_cast<long long>(t) * H;
    const i32 rows = B + 1;

    auto row_ptr = [&](i32 j) -> const T* {
        return (j < B) ? blocks + (static_cast<long long>(j) * block_rows + t) * H
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
    int H, int D, float eps)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const long long base = ((long long)t * H + h) * D;

    float acc = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float x = o[base + d];
        acc += x * x;
    }
    __shared__ float ssum;

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (threadIdx.x == 0) ssum = 0.f;
    __syncthreads();
    if ((threadIdx.x & (warpSize - 1)) == 0) atomicAdd(&ssum, acc);
    __syncthreads();

    const float scale = rsqrtf(ssum / static_cast<float>(D) + eps);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float gate = Elem<ElemT>::to_f32(g[base + d]);
        const float y = o[base + d] * scale * weight[d] * (1.f / (1.f + __expf(-gate)));
        out[base + d] = Elem<ElemT>::from_f32(y);
    }
}
}
