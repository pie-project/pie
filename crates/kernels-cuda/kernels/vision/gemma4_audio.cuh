#pragma once

#include "vision/gemma4_naive_kernels.cuh"

namespace pie::vision {

template <class T>
__global__ void k_matmul_bias(const T* x, const T* W, const T* b, T* y, int N, int K, int O) {
    int n = blockIdx.y * blockDim.y + threadIdx.y, o = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || o >= O) return;
    const T* xr = x + (long)n * K;
    const T* wr = W + (long)o * K;
    float a = b ? F(b[o]) : 0.f;
    for (int k = 0; k < K; k++) a += F(xr[k]) * F(wr[k]);
    y[(long)n * O + o] = Bf<T>(a);
}

template <class T>
__global__ void k_silu(const T* x, T* o, usize t) {
    usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i < t) { float v = F(x[i]); o[i] = Bf<T>(v / (1.f + __expf(-v))); }
}

template <class T>
__global__ void k_axpy(T* a, const T* b, float scale, usize t) {
    usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i < t) a[i] = Bf<T>(F(a[i]) + scale * F(b[i]));
}

template <class T>
__global__ void k_glu(const T* x, T* o, int N, int D) {
    int n = blockIdx.y * blockDim.y + threadIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || d >= D) return;
    float a = F(x[(long)n * 2 * D + d]), g = F(x[(long)n * 2 * D + D + d]);
    o[(long)n * D + d] = Bf<T>(a / (1.f + __expf(-g)));
}

template <class T>
__global__ void k_layernorm_relu(const T* x, const T* w, T* o, int R, int C, float eps) {
    int r = blockIdx.x;
    if (r >= R) return;
    const T* xr = x + (long)r * C;
    T* orow = o + (long)r * C;
    float m = 0;
    for (int c = threadIdx.x; c < C; c += blockDim.x) m += F(xr[c]);
    for (int s = warpSize / 2; s > 0; s >>= 1) m += __shfl_down_sync(0xffffffff, m, s);
    __shared__ float wm[32], wv[32], mean, inv;
    if ((threadIdx.x & 31) == 0) wm[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0;
        int nw = (blockDim.x + 31) / 32;
        for (int i = 0; i < nw; i++) t += wm[i];
        mean = t / C;
    }
    __syncthreads();
    float v = 0;
    for (int c = threadIdx.x; c < C; c += blockDim.x) { float d = F(xr[c]) - mean; v += d * d; }
    for (int s = warpSize / 2; s > 0; s >>= 1) v += __shfl_down_sync(0xffffffff, v, s);
    if ((threadIdx.x & 31) == 0) wv[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0;
        int nw = (blockDim.x + 31) / 32;
        for (int i = 0; i < nw; i++) t += wv[i];
        inv = rsqrtf(t / C + eps);
    }
    __syncthreads();
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float y = (F(xr[c]) - mean) * inv * (w ? F(w[c]) : 1.f);
        orow[c] = Bf<T>(y > 0.f ? y : 0.f);
    }
}

template <class T>
__global__ void k_conv2d_s2(const T* in, const T* W, T* out,
                            int IC, int Tin, int Fin, int OC, int To, int Fo) {
    int oc = blockIdx.z;
    int to = blockIdx.y * blockDim.y + threadIdx.y, fo = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= OC || to >= To || fo >= Fo) return;
    float acc = 0;
    for (int ic = 0; ic < IC; ic++) {
        const T* wk = W + (((long)oc * IC + ic) * 3) * 3;
        for (int kt = 0; kt < 3; kt++) for (int kf = 0; kf < 3; kf++) {
            int ti = to * 2 + kt - 1, fi = fo * 2 + kf - 1;
            if (ti < 0 || ti >= Tin || fi < 0 || fi >= Fin) continue;
            acc += F(in[((long)ic * Tin + ti) * Fin + fi]) * F(wk[kt * 3 + kf]);
        }
    }
    out[((long)oc * To + to) * Fo + fo] = Bf<T>(acc);
}

template <class T>
__global__ void k_chlast(const T* in, T* out, int OC, int To, int Fo) {
    int oc = blockIdx.z;
    int to = blockIdx.y * blockDim.y + threadIdx.y, fo = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= OC || to >= To || fo >= Fo) return;
    out[(((long)to * Fo + fo) * OC) + oc] = in[((long)oc * To + to) * Fo + fo];
}

template <class T>
__global__ void k_chfirst(const T* in, T* out, int OC, int To, int Fo) {
    int oc = blockIdx.z;
    int to = blockIdx.y * blockDim.y + threadIdx.y, fo = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= OC || to >= To || fo >= Fo) return;
    out[((long)oc * To + to) * Fo + fo] = in[(((long)to * Fo + fo) * OC) + oc];
}

template <class T>
__global__ void k_sscp_flatten(const T* in, T* out, int OC, int To, int Fo) {
    int to = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    int FoOC = Fo * OC;
    if (to >= To || j >= FoOC) return;
    int fo = j / OC, oc = j % OC;
    out[(long)to * FoOC + j] = in[((long)oc * To + to) * Fo + fo];
}

template <class T>
__global__ void k_qkv_scale(T* q, T* k, const T* pds, int N, int H, int hd,
                            float q_scale, float k_scale) {
    int n = blockIdx.y * blockDim.y + threadIdx.y, e = blockIdx.x * blockDim.x + threadIdx.x;
    int HD = H * hd;
    if (n >= N || e >= HD) return;
    int d = e % hd;
    float sp = logf(1.f + expf(F(pds[d])));
    q[(long)n * HD + e] = Bf<T>(F(q[(long)n * HD + e]) * q_scale * sp);
    k[(long)n * HD + e] = Bf<T>(F(k[(long)n * HD + e]) * k_scale);
}

template <class T>
__global__ void k_rel_pos_enc(T* pe, int P, int hidden) {
    int r = blockIdx.y * blockDim.y + threadIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= P || d >= hidden) return;
    int num_ts = hidden / 2;
    float log_inc = logf(10000.f / 1.f) / fmaxf((float)(num_ts - 1), 1.f);
    int m = d < num_ts ? d : (d - num_ts);
    float inv = expf((float)m * -log_inc);
    float pos = (float)((P - 1) - r);
    float t = pos * inv;
    pe[(long)r * hidden + d] = Bf<T>(d < num_ts ? sinf(t) : cosf(t));
}

template <class T>
__global__ void k_local_attn(const T* q, const T* k, const T* v,
                             const T* relk, T* out,
                             int N, int H, int hd, int P, float cap) {
    int head = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (head >= H || i >= N) return;
    float acc[256];
    for (int d = 0; d < hd; d++) acc[d] = 0.f;

    int lo = i - (P - 2);
    if (lo < 0) lo = 0;
    const T* qr = q + ((long)i * H + head) * hd;
    float mx = -1e30f;
    for (int j = lo; j <= i; j++) {
        const T* kr = k + ((long)j * H + head) * hd;
        const T* rr = relk + ((long)((P - 1) - (i - j)) * H + head) * hd;
        float s = 0;
        for (int d = 0; d < hd; d++) s += F(qr[d]) * (F(kr[d]) + F(rr[d]));
        s = cap * tanhf(s / cap);
        mx = fmaxf(mx, s);
    }
    float denom = 0;
    for (int j = lo; j <= i; j++) {
        const T* kr = k + ((long)j * H + head) * hd;
        const T* rr = relk + ((long)((P - 1) - (i - j)) * H + head) * hd;
        float s = 0;
        for (int d = 0; d < hd; d++) s += F(qr[d]) * (F(kr[d]) + F(rr[d]));
        s = cap * tanhf(s / cap);
        float w = __expf(s - mx);
        denom += w;
        const T* vr = v + ((long)j * H + head) * hd;
        for (int d = 0; d < hd; d++) acc[d] += w * F(vr[d]);
    }
    float inv = denom > 0.f ? 1.f / denom : 0.f;
    for (int d = 0; d < hd; d++) out[((long)i * H + head) * hd + d] = Bf<T>(acc[d] * inv);
}

}
