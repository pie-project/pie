#pragma once

#include "prelude/device.cuh"

namespace pie::vision {

template <class T>
__device__ __forceinline__ float F(T x) { return Elem<T>::to_f32(x); }
template <class T>
__device__ __forceinline__ T Bf(float x) { return Elem<T>::from_f32(x); }

template <class T>
__global__ void k_matmul(const T* x, const T* W, T* y, int N, int K, int O) {
    int n = blockIdx.y * blockDim.y + threadIdx.y, o = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || o >= O) return;
    const T* xr = x + (long)n * K;
    const T* wr = W + (long)o * K;
    float a = 0;
    for (int k = 0; k < K; k++) a += F(xr[k]) * F(wr[k]);
    y[(long)n * O + o] = Bf<T>(a);
}

template <class T>
__global__ void k_rms(const T* x, const T* w, T* o, int R, int D, float eps) {
    int r = blockIdx.x;
    if (r >= R) return;
    const T* xr = x + (long)r * D;
    T* orow = o + (long)r * D;
    float loc = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) { float v = F(xr[d]); loc += v * v; }
    for (int s = warpSize / 2; s > 0; s >>= 1) loc += __shfl_down_sync(0xffffffff, loc, s);
    __shared__ float warp[32], ss;
    if ((threadIdx.x & 31) == 0) warp[threadIdx.x >> 5] = loc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0;
        int nw = (blockDim.x + 31) / 32;
        for (int i = 0; i < nw; i++) t += warp[i];
        ss = rsqrtf(t / D + eps);
    }
    __syncthreads();
    float inv = ss;
    for (int d = threadIdx.x; d < D; d += blockDim.x) orow[d] = Bf<T>(F(xr[d]) * inv * (w ? F(w[d]) : 1.f));
}

template <class T>
__global__ void k_add(T* a, const T* b, usize n) {
    usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i < n) a[i] = Bf<T>(F(a[i]) + F(b[i]));
}

template <class T>
__global__ void k_f32_to_bf16(const float* a, T* o, usize n) {
    usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i < n) o[i] = Bf<T>(a[i]);
}

template <class T>
__global__ void k_gelu_erf(const T* x, T* o, usize n) {
    usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = F(x[i]);
    o[i] = Bf<T>(0.5f * v * (1.f + erff(v * 0.70710678118654752f)));
}

template <class T>
__global__ void k_layernorm(const T* x, const T* g, const T* bta, T* o, int R, int D, float eps) {
    int r = blockIdx.x;
    if (r >= R) return;
    const T* xr = x + (long)r * D;
    T* orow = o + (long)r * D;
    float sum = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) sum += F(xr[d]);
    for (int s = warpSize / 2; s > 0; s >>= 1) sum += __shfl_down_sync(0xffffffff, sum, s);
    __shared__ float warp[32], smean, svar;
    if ((threadIdx.x & 31) == 0) warp[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0;
        int nw = (blockDim.x + 31) / 32;
        for (int i = 0; i < nw; i++) t += warp[i];
        smean = t / D;
    }
    __syncthreads();
    float mean = smean, v = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) { float dx = F(xr[d]) - mean; v += dx * dx; }
    for (int s = warpSize / 2; s > 0; s >>= 1) v += __shfl_down_sync(0xffffffff, v, s);
    if ((threadIdx.x & 31) == 0) warp[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0;
        int nw = (blockDim.x + 31) / 32;
        for (int i = 0; i < nw; i++) t += warp[i];
        svar = rsqrtf(t / D + eps);
    }
    __syncthreads();
    float inv = svar;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float nrm = (F(xr[d]) - mean) * inv;
        orow[d] = Bf<T>(nrm * (g ? F(g[d]) : 1.f) + (bta ? F(bta[d]) : 0.f));
    }
}

}
