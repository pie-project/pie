#pragma once

// Hyper-connections: a row of the residual carries M streams of H, laid out
// [M][H] in one M·H-wide row.
//
// The manifold-constrained form (mHC, DeepSeek-V4 / GLM-5 Flash) reads the
// streams through a learned mix row: the f32-normed stream row projects to
// 2M + M² logits, which give a pre gate per stream (how much of each stream
// the layer reads), a post gate per stream (how much of the layer's output
// each stream takes) and an M×M combination made doubly stochastic by
// Sinkhorn. After the layer, stream j becomes
//
//   post[j]·x + Σ_i comb[i][j]·stream_i
//
// The gated-residual form (Qwen3.8 Flash Next) mixes the streams into the
// layer input by a sigmoid gate per entry and injects the layer's output
// back into every stream by a gate per stream.
//
// Every kernel works in f32 and rounds once on its store; a captured launch
// gates its rows on the window.

#include "prelude/device.hip"

namespace pie::elemwise {

constexpr int MAX_HC_MULT = 8;

__device__ __forceinline__ float sigmoid(float z) { return 1.f / (1.f + expf(-z)); }

__device__ __forceinline__ float fast_sigmoid(float z) { return 1.f / (1.f + __expf(-z)); }

// mixes[row][o] = Σ_d normed[row][d]·hc_fn[o][d]: one block per (row, o).
template <int BLOCK = 256>
__global__ __launch_bounds__(BLOCK) void hc_project(const float* __restrict__ normed,
                                                    const float* __restrict__ hc_fn,
                                                    float* __restrict__ mixes, int fan_in,
                                                    int mix_hc, const u32* __restrict__ win) {
    __shared__ float smem[BLOCK / kWave];
    const int o = static_cast<int>(blockIdx.x) % mix_hc;
    const int n = static_cast<int>(blockIdx.x) / mix_hc;
    const Window window{win};
    if (!window.live(n)) return;
    const long long row = window.plane(n);
    const float* src = normed + row * fan_in;
    const float* w = hc_fn + static_cast<long long>(o) * fan_in;

    float local = 0.f;
    if (fan_in % 4 == 0 && aligned16(src) && aligned16(w)) {
        for (int d = threadIdx.x; d < fan_in / 4; d += BLOCK) {
            const float4 a = reinterpret_cast<const float4*>(src)[d];
            const float4 b = reinterpret_cast<const float4*>(w)[d];
            local += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    } else {
        for (int d = threadIdx.x; d < fan_in; d += BLOCK) local += src[d] * w[d];
    }
    const float total = block_sum(local, smem);
    if (threadIdx.x == 0) mixes[row * mix_hc + o] = total;
}

// The gates of one row from its mix row, then the layer input
// x = Σ_i pre[i]·stream_i. Lands post[M] and comb[M][M] for `hc_fold`.
template <class T, int BLOCK = 256>
__global__ __launch_bounds__(BLOCK) void hc_gates(
    const float* __restrict__ mixes, const float* __restrict__ scale, const float* __restrict__ base,
    const T* __restrict__ residual, float* __restrict__ post_mix, float* __restrict__ comb_mix,
    T* __restrict__ layer_input, int M, int H, float hc_eps, float hc_post_alpha,
    int sinkhorn_iters, const u32* __restrict__ win) {
    static_assert(BLOCK >= MAX_HC_MULT * MAX_HC_MULT, "a thread per combination entry");
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const int tid = threadIdx.x;
    const float* mix = mixes + row * (2 * M + M * M);

    __shared__ float pre[MAX_HC_MULT];
    __shared__ float comb[MAX_HC_MULT * MAX_HC_MULT];

    if (tid < M) {
        pre[tid] = sigmoid(mix[tid] * scale[0] + base[tid]) + hc_eps;
        post_mix[row * M + tid] = sigmoid(mix[M + tid] * scale[1] + base[M + tid]) * hc_post_alpha;
    }
    if (tid < M * M) comb[tid] = mix[2 * M + tid] * scale[2] + base[2 * M + tid];
    lds_barrier();

    // Row softmax (+ eps), then a column normalisation: the first Sinkhorn
    // step. Each of the remaining iterations normalises rows, then columns.
    auto normalise_columns = [&]() {
        if (tid < M) {
            float sum = 0.f;
            for (int i = 0; i < M; ++i) sum += comb[i * M + tid];
            sum += hc_eps;
            for (int i = 0; i < M; ++i) comb[i * M + tid] /= sum;
        }
        lds_barrier();
    };
    if (tid < M) {
        float* r = comb + tid * M;
        float top = neg_inf();
        for (int j = 0; j < M; ++j) top = fmaxf(top, r[j]);
        float sum = 0.f;
        for (int j = 0; j < M; ++j) {
            r[j] = expf(r[j] - top);
            sum += r[j];
        }
        for (int j = 0; j < M; ++j) r[j] = r[j] / sum + hc_eps;
    }
    lds_barrier();
    normalise_columns();
    for (int iter = 1; iter < sinkhorn_iters; ++iter) {
        if (tid < M) {
            float* r = comb + tid * M;
            float sum = 0.f;
            for (int j = 0; j < M; ++j) sum += r[j];
            sum += hc_eps;
            for (int j = 0; j < M; ++j) r[j] /= sum;
        }
        lds_barrier();
        normalise_columns();
    }
    if (tid < M * M) comb_mix[row * M * M + tid] = comb[tid];

    const T* res = residual + row * M * H;
    T* out = layer_input + row * H;
    for (int h = tid; h < H; h += BLOCK) {
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < MAX_HC_MULT; ++i) {
            if (i < M) acc += pre[i] * to_f32(res[static_cast<long long>(i) * H + h]);
        }
        out[h] = from_f32<T>(acc);
    }
}

// The streams after the layer: out_j = post[j]·x + Σ_i comb[i][j]·stream_i,
// one thread per (row, h). `residual` and `out` may be one plane: a thread
// reads its column of every stream before it writes any.
template <class T>
__global__ void hc_fold(const T* __restrict__ x, const T* residual,
                        const float* __restrict__ post_mix, const float* __restrict__ comb_mix,
                        T* out, int N, int M, int H, const u32* __restrict__ win) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (M > MAX_HC_MULT || idx >= static_cast<long long>(N) * H) return;
    const int h = static_cast<int>(idx % H);
    const Window window{win};
    if (!window.live(idx / H)) return;
    const long long row = window.plane(idx / H);

    const float* comb = comb_mix + row * M * M;
    const float* post = post_mix + row * M;
    const float xh = to_f32(x[row * H + h]);
    const T* res = residual + row * M * H + h;
    T* dst = out + row * M * H + h;

    float r[MAX_HC_MULT];
#pragma unroll
    for (int i = 0; i < MAX_HC_MULT; ++i) {
        r[i] = i < M ? to_f32(res[static_cast<long long>(i) * H]) : 0.f;
    }
#pragma unroll
    for (int j = 0; j < MAX_HC_MULT; ++j) {
        if (j < M) {
            float acc = post[j] * xh;
#pragma unroll
            for (int i = 0; i < MAX_HC_MULT; ++i) {
                if (i < M) acc += comb[i * M + j] * r[i];
            }
            dst[static_cast<long long>(j) * H] = from_f32<T>(acc);
        }
    }
}

// The trunk's collapse before the head: y = Σ_i (sigmoid(logit_i) + eps)·stream_i.
template <class T, int BLOCK = 256>
__global__ __launch_bounds__(BLOCK) void hc_head_postprocess(
    const float* __restrict__ mixes, const float* __restrict__ scale, const float* __restrict__ base,
    const T* __restrict__ residual, T* __restrict__ out, int M, int H, float hc_eps,
    const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    __shared__ float gates[MAX_HC_MULT];
    if (threadIdx.x < M) {
        gates[threadIdx.x] = sigmoid(mixes[row * M + threadIdx.x] * scale[0] + base[threadIdx.x]) + hc_eps;
    }
    lds_barrier();

    const T* res = residual + row * M * H;
    T* dst = out + row * H;
    for (int h = threadIdx.x; h < H; h += BLOCK) {
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < MAX_HC_MULT; ++i) {
            if (i < M) acc += gates[i] * to_f32(res[static_cast<long long>(i) * H + h]);
        }
        dst[h] = from_f32<T>(acc);
    }
}

// Every stream of a row starts as the row: output[row][m][h] = input[row][h].
template <class T>
__global__ void hc_expand(const T* __restrict__ input, T* __restrict__ output, int N, int M, int H,
                          const u32* __restrict__ win) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long long>(N) * H) return;
    const int h = static_cast<int>(idx % H);
    const Window window{win};
    if (!window.live(idx / H)) return;
    const long long row = window.plane(idx / H);
    const T v = input[row * H + h];
    T* dst = output + row * M * H + h;
    for (int m = 0; m < M; ++m) dst[static_cast<long long>(m) * H] = v;
}

// The stream row widened to f32 and scaled by rsqrt(mean(x²) + eps), no
// weight: the mix projection's input.
template <class T, int BLOCK = 256>
__global__ __launch_bounds__(BLOCK) void hc_rmsnorm_f32(const T* __restrict__ input,
                                                        float* __restrict__ output, int dim,
                                                        float eps, const u32* __restrict__ win) {
    __shared__ float smem[BLOCK / kWave];
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const T* src = input + row * dim;
    float* dst = output + row * dim;
    float local = 0.f;
    for (int d = threadIdx.x; d < dim; d += BLOCK) {
        const float v = to_f32(src[d]);
        local += v * v;
    }
    const float s = rsqrtf(block_sum(local, smem) / static_cast<float>(dim) + eps);
    for (int d = threadIdx.x; d < dim; d += BLOCK) dst[d] = to_f32(src[d]) * s;
}

// The layer input from the streams: y[h] = mean_s(normed_s[h]·sigmoid(gate_s[h])).
template <class T, int BLOCK = 256>
__global__ __launch_bounds__(BLOCK) void hc_mix(const T* __restrict__ gates,
                                                const T* __restrict__ normed, T* __restrict__ y,
                                                int M, int H, const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const T* g = gates + row * M * H;
    const T* v = normed + row * M * H;
    T* dst = y + row * H;
    for (int h = threadIdx.x; h < H; h += BLOCK) {
        float acc = 0.f;
        for (int s = 0; s < M; ++s) {
            const long long at = static_cast<long long>(s) * H + h;
            acc += to_f32(v[at]) / (1.f + __expf(-to_f32(g[at])));
        }
        dst[h] = from_f32<T>(acc / static_cast<float>(M));
    }
}

// The layer's output back into every stream: hyper_s += 2·sigmoid(gate_s/M)·o.
template <class T, int BLOCK = 256>
__global__ __launch_bounds__(BLOCK) void hc_inject(const T* __restrict__ o,
                                                   const T* __restrict__ gates,
                                                   T* __restrict__ hyper, int M, int H,
                                                   const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const T* orow = o + row * H;
    T* hr = hyper + row * M * H;
    __shared__ float g[MAX_HC_MULT];
    if (threadIdx.x < M) {
        g[threadIdx.x] = 2.f * fast_sigmoid(to_f32(gates[row * M + threadIdx.x]) / static_cast<float>(M));
    }
    lds_barrier();
    for (int s = 0; s < M; ++s) {
        for (int h = threadIdx.x; h < H; h += BLOCK) {
            const long long at = static_cast<long long>(s) * H + h;
            hr[at] = from_f32<T>(to_f32(hr[at]) + g[s] * to_f32(orow[h]));
        }
    }
}

// The per-layer-embedding gate per stream: one block per (row, s) scores
// d = (key_s · query_s)/√H, damps it to sign(d)·√max(|d|, 1e-6), and lands
// y_s = sigmoid(damped)·value.
template <class T, int BLOCK = 256>
__global__ __launch_bounds__(BLOCK) void ple_gate(const T* __restrict__ key,
                                                  const T* __restrict__ query,
                                                  const T* __restrict__ value, T* __restrict__ y,
                                                  int M, int H, const u32* __restrict__ win) {
    __shared__ float smem[BLOCK / kWave];
    const int n = static_cast<int>(blockIdx.x) / M;
    const int s = static_cast<int>(blockIdx.x) - n * M;
    const Window window{win};
    if (!window.live(n)) return;
    const long long row = window.plane(n);
    const long long at = (row * M + s) * H;
    const T* kr = key + at;
    const T* qr = query + at;
    const T* vr = value + row * H;
    T* yr = y + at;

    float local = 0.f;
    for (int i = threadIdx.x; i < H; i += BLOCK) local += to_f32(kr[i]) * to_f32(qr[i]);
    const float dot = block_sum(local, smem) * rsqrtf(static_cast<float>(H));
    const float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
    const float damped = dot > 0.f ? mag : (dot < 0.f ? -mag : 0.f);
    const float gate = fast_sigmoid(damped);
    for (int i = threadIdx.x; i < H; i += BLOCK) yr[i] = from_f32<T>(gate * to_f32(vr[i]));
}

}  // namespace pie::elemwise
