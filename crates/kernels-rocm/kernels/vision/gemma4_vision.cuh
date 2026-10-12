#pragma once

#include "vision/gemma4_naive_kernels.cuh"

namespace pie::vision {

template <class T>
__global__ void k_scale(const T* p, T* o, usize t) {
    usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i < t) o[i] = Bf<T>(2.f * (F(p[i]) - 0.5f));
}

template <class T>
__global__ void k_addpos_grid2d(T* y, const T* tb, const float* pos, int N, int O, int P) {
    int n = blockIdx.y * blockDim.y + threadIdx.y, o = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || o >= O) return;
    long x = (long)llrintf(pos[2L * n]), yy = (long)llrintf(pos[2L * n + 1]);
    if (x < 0) x = 0;
    if (yy < 0) yy = 0;
    y[(long)n * O + o] = Bf<T>(F(y[(long)n * O + o]) + F(tb[(0L * P + x) * O + o]) + F(tb[(1L * P + yy) * O + o]));
}

template <class T>
__global__ void k_rope_axial2d(T* q, const float* pos, int N, int H, float theta) {
    int n = blockIdx.z, head = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || head >= H || c >= 16) return;
    T* v = q + (((long)n * H + head) * 64);
    float px = pos[2L * n], py = pos[2L * n + 1];
    float invf = powf(theta, -(float)c / 16.f);
    float cx = cosf(px * invf), sx = sinf(px * invf), cy = cosf(py * invf), sy = sinf(py * invf);
    float a = F(v[c]), b = F(v[c + 16]);
    v[c] = Bf<T>(a * cx - b * sx);
    v[c + 16] = Bf<T>(b * cx + a * sx);
    float e = F(v[32 + c]), f = F(v[48 + c]);
    v[32 + c] = Bf<T>(e * cy - f * sy);
    v[48 + c] = Bf<T>(f * cy + e * sy);
}

template <class T>
__global__ void k_qk(const T* q, const T* k, float* s, int N, int H, int head, float scale) {
    int i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= N) return;
    const T* qi = q + ((long)i * H + head) * 64;
    const T* kj = k + ((long)j * H + head) * 64;
    float a = 0;
    for (int d = 0; d < 64; d++) a += F(qi[d]) * F(kj[d]);
    s[(long)i * N + j] = a * scale;
}

template <class T>
__global__ void k_softmax(float* s, int N) {
    int i = blockIdx.x;
    if (i >= N) return;
    float* r = s + (long)i * N;
    float mx = -1e30f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) mx = fmaxf(mx, r[j]);
    for (int o = warpSize / 2; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, o));
    __shared__ float wm[32], wsv[32], smx, ssum;
    if ((threadIdx.x & 31) == 0) wm[threadIdx.x >> 5] = mx;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -1e30f;
        int nw = (blockDim.x + 31) / 32;
        for (int i2 = 0; i2 < nw; i2++) m = fmaxf(m, wm[i2]);
        smx = m;
    }
    __syncthreads();
    float sm = 0;
    for (int j = threadIdx.x; j < N; j += blockDim.x) { float e = __expf(r[j] - smx); r[j] = e; sm += e; }
    for (int o = warpSize / 2; o > 0; o >>= 1) sm += __shfl_down_sync(0xffffffff, sm, o);
    if ((threadIdx.x & 31) == 0) wsv[threadIdx.x >> 5] = sm;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0;
        int nw = (blockDim.x + 31) / 32;
        for (int i2 = 0; i2 < nw; i2++) t += wsv[i2];
        ssum = t;
    }
    __syncthreads();
    float inv = 1.f / ssum;
    for (int j = threadIdx.x; j < N; j += blockDim.x) r[j] *= inv;
}

template <class T>
__global__ void k_av(const float* s, const T* v, T* o, int N, int H, int head) {
    int n = blockIdx.y * blockDim.y + threadIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || d >= 64) return;
    const float* sr = s + (long)n * N;
    float a = 0;
    for (int j = 0; j < N; j++) a += sr[j] * F(v[((long)j * H + head) * 64 + d]);
    o[((long)n * H + head) * 64 + d] = Bf<T>(a);
}

template <class T>
__global__ void k_gelu_mul(const T* g, const T* u, T* o, usize t) {
    usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i < t) {
        float x = F(g[i]);
        float gl = 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        o[i] = Bf<T>(gl * F(u[i]));
    }
}

template <class T>
__global__ void k_pool(const T* h, const int* grp, float* o, int N, int D, float k2) {
    int n = blockIdx.y * blockDim.y + threadIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || d >= D) return;
    atomicAdd(&o[(long)grp[n] * D + d], F(h[(long)n * D + d]) / k2);
}

template <class T>
__global__ void k_pool_finish(const float* in, T* o, float s, usize t) {
    usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i < t) o[i] = Bf<T>(in[i] * s);
}

}
