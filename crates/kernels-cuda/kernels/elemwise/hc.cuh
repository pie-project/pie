#pragma once

#include "prelude/device.cuh"

namespace pie::elemwise {

constexpr int MAX_HC_MULT = 8;

template <class T, int BLOCK = 256>
__global__ void hc_gates(
    const float* __restrict__ mixes,
    const float* __restrict__ scale,
    const float* __restrict__ base,
    const T* __restrict__ residual,
    float* __restrict__ post_mix,
    float* __restrict__ comb_mix,
    T* __restrict__ layer_input,
    int M,
    int H,
    float hc_eps,
    float hc_post_alpha,
    int sinkhorn_iters)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;

    const int mix_hc = M * 2 + M * M;
    const float* row = mixes + static_cast<long long>(n) * mix_hc;

    __shared__ float pre[MAX_HC_MULT];
    __shared__ float post[MAX_HC_MULT];
    __shared__ float comb[MAX_HC_MULT * MAX_HC_MULT];

    if (tid < M) {

        const float logit = row[tid] * scale[0] + base[tid];
        pre[tid] = 1.f / (1.f + expf(-logit)) + hc_eps;
    }
    if (tid < M) {

        const float logit = row[M + tid] * scale[1] + base[M + tid];
        post[tid] = 1.f / (1.f + expf(-logit)) * hc_post_alpha;
        post_mix[static_cast<long long>(n) * M + tid] = post[tid];
    }
    __syncthreads();

    if (tid < M * M) {
        const float logit = row[2 * M + tid] * scale[2] + base[2 * M + tid];
        comb[tid] = logit;
    }
    __syncthreads();

    if (tid < M) {
        float max_v = -flt_max();
        for (int j = 0; j < M; ++j)
            max_v = fmaxf(max_v, comb[tid * M + j]);
        float sum = 0.f;
        for (int j = 0; j < M; ++j) {
            comb[tid * M + j] = expf(comb[tid * M + j] - max_v);
            sum += comb[tid * M + j];
        }
        for (int j = 0; j < M; ++j)
            comb[tid * M + j] = comb[tid * M + j] / sum + hc_eps;
    }
    __syncthreads();

    if (tid < M) {
        float col_sum = 0.f;
        for (int i = 0; i < M; ++i) col_sum += comb[i * M + tid];
        col_sum += hc_eps;
        for (int i = 0; i < M; ++i)
            comb[i * M + tid] = comb[i * M + tid] / col_sum;
    }
    __syncthreads();

    for (int iter = 0; iter < sinkhorn_iters - 1; ++iter) {

        if (tid < M) {
            float row_sum = 0.f;
            for (int j = 0; j < M; ++j) row_sum += comb[tid * M + j];
            row_sum += hc_eps;
            for (int j = 0; j < M; ++j)
                comb[tid * M + j] = comb[tid * M + j] / row_sum;
        }
        __syncthreads();

        if (tid < M) {
            float col_sum = 0.f;
            for (int i = 0; i < M; ++i) col_sum += comb[i * M + tid];
            col_sum += hc_eps;
            for (int i = 0; i < M; ++i)
                comb[i * M + tid] = comb[i * M + tid] / col_sum;
        }
        __syncthreads();
    }

    if (tid < M * M) {
        comb_mix[static_cast<long long>(n) * M * M + tid] = comb[tid];
    }
    __syncthreads();

    const T* res_n = residual + static_cast<long long>(n) * M * H;
    T* out = layer_input + static_cast<long long>(n) * H;

    for (int h = tid; h < H; h += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < M; ++i) {
            acc += pre[i] * Elem<T>::to_f32(res_n[i * H + h]);
        }
        out[h] = Elem<T>::from_f32(acc);
    }
}

template <class T>
__global__ void hc_fold(
    const T* __restrict__ x,
    const T* residual,
    const float* __restrict__ post_mix,
    const float* __restrict__ comb_mix,
    T* out,
    int N,
    int M,
    int H)
{
    if (M > MAX_HC_MULT) return;
    const long long idx =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long long>(N) * H) return;

    const int h = static_cast<int>(idx % H);
    const int n = static_cast<int>(idx / H);

    const float* comb_n = comb_mix + static_cast<long long>(n) * M * M;
    const float* post_n = post_mix + static_cast<long long>(n) * M;
    const float x_h = Elem<T>::to_f32(x[static_cast<long long>(n) * H + h]);
    const T* res_n = residual + static_cast<long long>(n) * M * H;

    float r[MAX_HC_MULT];
    for (int i = 0; i < M; ++i) {
        r[i] = Elem<T>::to_f32(res_n[static_cast<long long>(i) * H + h]);
    }

    T* out_n = out + static_cast<long long>(n) * M * H;

    for (int j = 0; j < M; ++j) {
        float acc = post_n[j] * x_h;
        for (int i = 0; i < M; ++i) {
            acc += comb_n[i * M + j] * r[i];
        }
        out_n[static_cast<long long>(j) * H + h] = Elem<T>::from_f32(acc);
    }
}

template <class T, int BLOCK = 256>
__global__ void hc_head_postprocess(
    const float* __restrict__ mixes,
    const float* __restrict__ scale,
    const float* __restrict__ base,
    const T* __restrict__ residual,
    T* __restrict__ out,
    int M,
    int H,
    float hc_eps)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float gates[MAX_HC_MULT];

    if (tid < M) {
        const float logit = mixes[static_cast<long long>(n) * M + tid] * scale[0] + base[tid];
        gates[tid] = 1.f / (1.f + expf(-logit)) + hc_eps;
    }
    __syncthreads();

    const T* res_n = residual + static_cast<long long>(n) * M * H;
    T* out_n = out + static_cast<long long>(n) * H;

    for (int h = tid; h < H; h += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < M; ++i) {
            acc += gates[i] * Elem<T>::to_f32(res_n[i * H + h]);
        }
        out_n[h] = Elem<T>::from_f32(acc);
    }
}

template <class T>
__global__ void hc_expand(
    const T* __restrict__ input,
    T* __restrict__ output,
    int N,
    int M,
    int H)
{
    const long long idx =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long long>(N) * H) return;
    const int n = static_cast<int>(idx / H);
    const int h = static_cast<int>(idx % H);

    const T val = input[static_cast<long long>(n) * H + h];
    for (int m = 0; m < M; ++m) {
        output[static_cast<long long>(n) * M * H + m * H + h] = val;
    }
}

template <class T, int BLOCK = 256>
__global__ void hc_rmsnorm_f32(
    const T* __restrict__ input,
    float* __restrict__ output,
    int dim,
    float eps)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const T* row = input + static_cast<long long>(n) * dim;
    float* out = output + static_cast<long long>(n) * dim;

    float local_sum = 0.f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float v = Elem<T>::to_f32(row[d]);
        local_sum += v * v;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);

    __shared__ float warp_sums[BLOCK / 32];
    if ((tid & 31) == 0) warp_sums[tid >> 5] = local_sum;
    __syncthreads();

    __shared__ float scale;
    if (tid == 0) {
        float total = 0.f;
        const int nwarps = (blockDim.x + 31) / 32;
        for (int w = 0; w < nwarps; ++w) total += warp_sums[w];
        scale = rsqrtf(total / dim + eps);
    }
    __syncthreads();

    const float s = scale;
    for (int d = tid; d < dim; d += blockDim.x) {
        out[d] = Elem<T>::to_f32(row[d]) * s;
    }
}

template <class T>
__global__ void attn_sink_correction(
    T* __restrict__ out,
    const float* __restrict__ lse,
    const float* __restrict__ sink,
    int num_heads,
    int head_dim)
{
    const int n = blockIdx.x;
    const int h = blockIdx.y;
    const float s = 1.0f / (1.0f + expf(sink[h] - lse[n * num_heads + h]));
    T* row = out + (static_cast<long long>(n) * num_heads + h) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        row[d] = Elem<T>::from_f32(Elem<T>::to_f32(row[d]) * s);
    }
}

template <class T>
__global__ void per_head_rmsnorm(
    T* __restrict__ q,
    int head_dim,
    float eps)
{

    const int n = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int num_heads = gridDim.y;

    T* row = q + (static_cast<long long>(n) * num_heads + h) * head_dim;

    float local_sum = 0.f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        const float v = Elem<T>::to_f32(row[d]);
        local_sum += v * v;
    }
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, off);

    __shared__ float scale;
    __shared__ float reduce_buf[32];
    if ((tid & 31) == 0) reduce_buf[tid >> 5] = local_sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? reduce_buf[tid] : 0.f;
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xFFFFFFFF, v, off);
        if (tid == 0) scale = rsqrtf(v / static_cast<float>(head_dim) + eps);
    }
    __syncthreads();

    const float s = scale;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        row[d] = Elem<T>::from_f32(Elem<T>::to_f32(row[d]) * s);
    }
}

}
