#pragma once

// Hyper-connections: a residual row carries M streams of H ([M][H]). mHC
// (DeepSeek-V4, GLM-5 Flash) projects the f32-normed row to 2M + M² logits:
// pre and post gates a stream and an M×M mix made doubly stochastic by
// Sinkhorn; stream j becomes post[j]·x + Σ_i comb[i][j]·stream_i. The
// gated-residual form (Qwen3.8 Flash Next) mixes and injects by sigmoid
// gates. f32 throughout, one rounding a store.

#include "prim/device.hip"
#include "prim/mfma.hip"

namespace pie::elemwise {

constexpr int MAX_HC_MULT = 8;

__device__ __forceinline__ float sigmoid(float z) { return 1.f / (1.f + expf(-z)); }

__device__ __forceinline__ float fast_sigmoid(float z) { return 1.f / (1.f + __expf(-z)); }

// The streams' kernels split each row along H: block (n, c) takes elements
// [c·8·BLOCK, (c+1)·8·BLOCK) of row n, every thread a pack of 8 when the
// launcher found the row 16-byte aligned (VEC) and otherwise 8 elements
// BLOCK apart. M (the stream count) is a template parameter, so the stream
// loops unroll and every stream's pack is in flight at once.

template <class T>
__device__ __forceinline__ void load8(const T* p, float (&v)[8]) {
    const Pack8<T> q = *reinterpret_cast<const Pack8<T>*>(p);
#pragma unroll
    for (int k = 0; k < 8; ++k) v[k] = to_f32(q.v[k]);
}

template <class T>
__device__ __forceinline__ void store8(T* p, const float (&v)[8]) {
    Pack8<T> q;
#pragma unroll
    for (int k = 0; k < 8; ++k) q.v[k] = from_f32<T>(v[k]);
    *reinterpret_cast<Pack8<T>*>(p) = q;
}

// The column h of chunk `chunk` the thread's k-th element is, and whether it
// is one: a pack's k-th lane when VEC, otherwise every BLOCK-th column.
template <int BLOCK, bool VEC>
__device__ __forceinline__ int column(int chunk, int k) {
    return VEC ? (chunk * BLOCK + static_cast<int>(threadIdx.x)) * 8 + k
               : chunk * BLOCK * 8 + k * BLOCK + static_cast<int>(threadIdx.x);
}

// out[h] = Σ_i g[i]·stream_i[h] over the thread's columns of one row.
template <class T, int M, int BLOCK, bool VEC>
__device__ __forceinline__ void combine_streams(const float (&g)[M], const T* __restrict__ res,
                                                T* __restrict__ out, int H, int chunk) {
    if constexpr (VEC) {
        const int h = column<BLOCK, VEC>(chunk, 0);
        if (h >= H) return;
        float v[M][8];
#pragma unroll
        for (int i = 0; i < M; ++i) load8(res + static_cast<long long>(i) * H + h, v[i]);
        float acc[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            acc[k] = 0.f;
#pragma unroll
            for (int i = 0; i < M; ++i) acc[k] += g[i] * v[i][k];
        }
        store8(out + h, acc);
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int h = column<BLOCK, VEC>(chunk, k);
            if (h < H) {
                float acc = 0.f;
#pragma unroll
                for (int i = 0; i < M; ++i) acc += g[i] * to_f32(res[static_cast<long long>(i) * H + h]);
                out[h] = from_f32<T>(acc);
            }
        }
    }
}

// mixes[row][o] = Σ_d normed[row][d]·hc_fn[o][d]: one block per (row, o),
// the fallback for a contraction the tiled form does not take.
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

// The mix projection as a K-sliced f32 MFMA product: block (s, t) takes
// rows [16·NT·t, +16·NT) of K slice s, its wavefronts interleaved 64 k a
// step, A and B read alike so the product matches. The tiles add in wave
// order; a slice's sums land in part[s] for `hc_project_sum` (one slice:
// the mixes). K a multiple of 64, operands 16-byte aligned.
template <int O, int NT, int WAVES>
__global__ __launch_bounds__(WAVES * 64) void hc_project_tiled(const float* __restrict__ normed,
                                                        const float* __restrict__ hc_fn,
                                                        float* __restrict__ part,
                                                        float* __restrict__ mixes, int N, int K,
                                                        int per, const u32* __restrict__ win) {
    constexpr int OT = (O + 15) / 16;
    __shared__ mfma_cd tiles[WAVES - 1][NT][OT][kWave];
    const int n0 = static_cast<int>(blockIdx.y) * 16 * NT;
    const Window window{win};
    if (!window.live(n0)) return;
    const int l = lane(), w = static_cast<int>(threadIdx.x) / kWave, r = l % 16, g = l / 16;
    // A row past the launch or the window reads the tile's first row; its
    // sums are never landed.
    bool lands[NT];
    const float* x[NT];
#pragma unroll
    for (int b = 0; b < NT; ++b) {
        const int n = n0 + 16 * b + r;
        lands[b] = n < N && window.live(n);
        x[b] = normed + window.plane(lands[b] ? n : n0) * K + 4 * g;
    }
    const float* a[OT];
#pragma unroll
    for (int t = 0; t < OT; ++t) a[t] = hc_fn + static_cast<long long>(min(16 * t + r, O - 1)) * K + 4 * g;

    mfma_cd acc[NT][OT];
#pragma unroll
    for (int b = 0; b < NT; ++b) {
#pragma unroll
        for (int t = 0; t < OT; ++t) acc[b][t] = mfma_zero();
    }
    const int begin = static_cast<int>(blockIdx.x) * per;
    const int end = min(K, begin + per);
    for (int kb = begin + 64 * w; kb < end; kb += 64 * WAVES) {
        float4 xv[NT][4];
        float4 av[OT][4];
#pragma unroll
        for (int b = 0; b < NT; ++b) {
#pragma unroll
            for (int u = 0; u < 4; ++u) xv[b][u] = *reinterpret_cast<const float4*>(x[b] + kb + 16 * u);
        }
#pragma unroll
        for (int t = 0; t < OT; ++t) {
#pragma unroll
            for (int u = 0; u < 4; ++u) av[t][u] = *reinterpret_cast<const float4*>(a[t] + kb + 16 * u);
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
#pragma unroll
            for (int b = 0; b < NT; ++b) {
#pragma unroll
                for (int t = 0; t < OT; ++t) {
                    acc[b][t] = mfma4_f32(av[t][u].x, xv[b][u].x, acc[b][t]);
                    acc[b][t] = mfma4_f32(av[t][u].y, xv[b][u].y, acc[b][t]);
                    acc[b][t] = mfma4_f32(av[t][u].z, xv[b][u].z, acc[b][t]);
                    acc[b][t] = mfma4_f32(av[t][u].w, xv[b][u].w, acc[b][t]);
                }
            }
        }
    }
    if (w > 0) {
#pragma unroll
        for (int b = 0; b < NT; ++b) {
#pragma unroll
            for (int t = 0; t < OT; ++t) tiles[w - 1][b][t][l] = acc[b][t];
        }
    }
    lds_barrier();
    if (w > 0) return;
#pragma unroll
    for (int b = 0; b < NT; ++b) {
        if (!lands[b]) continue;
        const int n = n0 + 16 * b + r;
#pragma unroll
        for (int v = 0; v < WAVES - 1; ++v) {
#pragma unroll
            for (int t = 0; t < OT; ++t) acc[b][t] += tiles[v][b][t][l];
        }
        // The lane's cells: outputs 16t + 4g + e of row n.
        float* dst = part != nullptr
                         ? part + (static_cast<long long>(blockIdx.x) * N + n) * O
                         : mixes + window.plane(n) * O;
#pragma unroll
        for (int t = 0; t < OT; ++t) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int o = 16 * t + 4 * g + e;
                if (o < O) dst[o] = acc[b][t][e];
            }
        }
    }
}

// The K slices' sums in slice order: mixes[row][o] = Σ_s part[s][n][o],
// every slice's load in flight at once.
template <int O, int S>
__global__ void hc_project_sum(const float* __restrict__ part, float* __restrict__ mixes, int N,
                               const u32* __restrict__ win) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= N * O) return;
    const int n = idx / O;
    const Window window{win};
    if (!window.live(n)) return;
    const long long stride = static_cast<long long>(N) * O;
    const float* p = part + idx;
    float v[S];
#pragma unroll
    for (int s = 0; s < S; ++s) v[s] = p[s * stride];
    float acc = 0.f;
#pragma unroll
    for (int s = 0; s < S; ++s) acc += v[s];
    mixes[window.plane(n) * O + (idx - n * O)] = acc;
}

// The mix projection of the normed stream row, from the streams: the norm
// has no weight, so mixes = (x·fnᵀ)·rsqrt(mean(x²) + eps), and the f32 row
// is never written. The tiling is `hc_project_tiled`'s over 2-byte streams:
// a lane reads 8 k of its row a 16-byte pack and feeds them to eight steps,
// and alongside the products sums its squares. Block (s, t) lands its slice's
// sums in part[s][n][o] and its rows' squares in part[s][n][O], a row of
// O + 1, for `hc_norm_project_sum`. K is a multiple of 64 and every operand 16-byte
// aligned (the launcher's check).
template <class T, int O, int NT, int WAVES>
__global__ __launch_bounds__(WAVES * 64) void hc_norm_project_tiled(
    const T* __restrict__ streams, const float* __restrict__ hc_fn, float* __restrict__ part,
    int N, int K, int per, const u32* __restrict__ win) {
    constexpr int OT = (O + 15) / 16;
    __shared__ mfma_cd tiles[WAVES - 1][NT][OT][kWave];
    __shared__ float squares[WAVES - 1][NT][16];
    const int n0 = static_cast<int>(blockIdx.y) * 16 * NT;
    const Window window{win};
    if (!window.live(n0)) return;
    const int l = lane(), w = static_cast<int>(threadIdx.x) / kWave, r = l % 16, g = l / 16;
    bool lands[NT];
    const T* x[NT];
#pragma unroll
    for (int b = 0; b < NT; ++b) {
        const int n = n0 + 16 * b + r;
        lands[b] = n < N && window.live(n);
        x[b] = streams + window.plane(lands[b] ? n : n0) * K + 8 * g;
    }
    const float* a[OT];
#pragma unroll
    for (int t = 0; t < OT; ++t) a[t] = hc_fn + static_cast<long long>(min(16 * t + r, O - 1)) * K + 8 * g;

    mfma_cd acc[NT][OT];
    float sq[NT];
#pragma unroll
    for (int b = 0; b < NT; ++b) {
        sq[b] = 0.f;
#pragma unroll
        for (int t = 0; t < OT; ++t) acc[b][t] = mfma_zero();
    }
    const int begin = static_cast<int>(blockIdx.x) * per;
    const int end = min(K, begin + per);
    for (int kb = begin + 64 * w; kb < end; kb += 64 * WAVES) {
        float xv[NT][2][8];
        float4 av[OT][2][2];
#pragma unroll
        for (int b = 0; b < NT; ++b) {
#pragma unroll
            for (int u = 0; u < 2; ++u) load8(x[b] + kb + 32 * u, xv[b][u]);
        }
#pragma unroll
        for (int t = 0; t < OT; ++t) {
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                av[t][u][0] = *reinterpret_cast<const float4*>(a[t] + kb + 32 * u);
                av[t][u][1] = *reinterpret_cast<const float4*>(a[t] + kb + 32 * u + 4);
            }
        }
#pragma unroll
        for (int u = 0; u < 2; ++u) {
#pragma unroll
            for (int b = 0; b < NT; ++b) {
#pragma unroll
                for (int e = 0; e < 8; ++e) sq[b] += xv[b][u][e] * xv[b][u][e];
#pragma unroll
                for (int t = 0; t < OT; ++t) {
#pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        const float4 wv = av[t][u][q];
                        acc[b][t] = mfma4_f32(wv.x, xv[b][u][4 * q + 0], acc[b][t]);
                        acc[b][t] = mfma4_f32(wv.y, xv[b][u][4 * q + 1], acc[b][t]);
                        acc[b][t] = mfma4_f32(wv.z, xv[b][u][4 * q + 2], acc[b][t]);
                        acc[b][t] = mfma4_f32(wv.w, xv[b][u][4 * q + 3], acc[b][t]);
                    }
                }
            }
        }
    }
    // A row's squares over the wavefront's four lane groups, then over the
    // waves in wave order.
#pragma unroll
    for (int b = 0; b < NT; ++b) {
        sq[b] += __shfl_xor(sq[b], 16, kWave);
        sq[b] += __shfl_xor(sq[b], 32, kWave);
    }
    if (w > 0) {
#pragma unroll
        for (int b = 0; b < NT; ++b) {
#pragma unroll
            for (int t = 0; t < OT; ++t) tiles[w - 1][b][t][l] = acc[b][t];
            if (g == 0) squares[w - 1][b][r] = sq[b];
        }
    }
    lds_barrier();
    if (w > 0) return;
#pragma unroll
    for (int b = 0; b < NT; ++b) {
        if (!lands[b]) continue;
        const int n = n0 + 16 * b + r;
#pragma unroll
        for (int v = 0; v < WAVES - 1; ++v) {
            sq[b] += squares[v][b][r];
#pragma unroll
            for (int t = 0; t < OT; ++t) acc[b][t] += tiles[v][b][t][l];
        }
        float* dst = part + (static_cast<long long>(blockIdx.x) * N + n) * (O + 1);
        if (g == 0) dst[O] = sq[b];
#pragma unroll
        for (int t = 0; t < OT; ++t) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int o = 16 * t + 4 * g + e;
                if (o < O) dst[o] = acc[b][t][e];
            }
        }
    }
}

// The K slices' sums in slice order, scaled by the row's norm:
// mixes[row][o] = Σ_s part[s][n][o] · rsqrt(Σ_s part[s][n][O] / K + eps),
// a block a row and a thread a column.
template <int O, int S>
__global__ __launch_bounds__((O + kWave) / kWave * kWave) void hc_norm_project_sum(
    const float* __restrict__ part, float* __restrict__ mixes, int N, int K, float eps,
    const u32* __restrict__ win) {
    __shared__ float squares;
    const int n = static_cast<int>(blockIdx.x), o = static_cast<int>(threadIdx.x);
    const Window window{win};
    if (!window.live(n)) return;
    const long long stride = static_cast<long long>(N) * (O + 1);
    const float* p = part + static_cast<long long>(n) * (O + 1) + min(o, O);
    float v[S];
#pragma unroll
    for (int s = 0; s < S; ++s) v[s] = p[s * stride];
    float acc = 0.f;
#pragma unroll
    for (int s = 0; s < S; ++s) acc += v[s];
    if (o == O) squares = acc;
    lds_barrier();
    if (o < O) mixes[window.plane(n) * O + o] = acc * rsqrtf(squares / static_cast<float>(K) + eps);
}

// Sinkhorn on one wavefront: lane t < M² holds combination entry
// (t / M, t % M), a row or column sum gathers its M entries in index order,
// and every lane past M² mirrors the last entry. Row softmax (+ eps), then a
// column normalisation, is the first step; each of the remaining
// iterations normalises rows, then columns.
template <int M>
__device__ __forceinline__ float sinkhorn(float c, float eps, int iters) {
    const int t = min(lane(), M * M - 1);
    const int i = t / M, j = t - (t / M) * M;
    auto row_sum = [&](float v) {
        float sum = 0.f;
#pragma unroll
        for (int jj = 0; jj < M; ++jj) sum += __shfl(v, i * M + jj, kWave);
        return sum;
    };
    auto normalise_column = [&]() {
        float sum = 0.f;
#pragma unroll
        for (int ii = 0; ii < M; ++ii) sum += __shfl(c, ii * M + j, kWave);
        c /= sum + eps;
    };
    float top = __shfl(c, i * M, kWave);
#pragma unroll
    for (int jj = 1; jj < M; ++jj) top = fmaxf(top, __shfl(c, i * M + jj, kWave));
    const float e = expf(c - top);
    c = e / row_sum(e) + eps;
    normalise_column();
    for (int iter = 1; iter < iters; ++iter) {
        c /= row_sum(c) + eps;
        normalise_column();
    }
    return c;
}

// The gates of one row from its mix row, then the layer input
// x = Σ_i pre[i]·stream_i. Block (n, 0) lands post[M] and comb[M][M] for
// `hc_fold` on its first wavefront; block (n, 1 + c) lands chunk c of x,
// its pre gates its own. Without a combine (COMB false, the independent
// form) the mix row is the pre and post logits alone, the post gate carries
// the eps the pre gate does, and the combine landed is the identity.
template <class T, int M, int BLOCK, bool VEC, bool COMB>
__global__ __launch_bounds__(BLOCK) void hc_gates(
    const float* __restrict__ mixes, const float* __restrict__ scale, const float* __restrict__ base,
    const T* __restrict__ residual, float* __restrict__ post_mix, float* __restrict__ comb_mix,
    T* __restrict__ layer_input, int H, float hc_eps, float hc_post_alpha, int sinkhorn_iters,
    const u32* __restrict__ win) {
    static_assert(M >= 1 && M <= MAX_HC_MULT && M * M <= kWave, "a lane per combination entry");
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    constexpr int kMixRow = COMB ? 2 * M + M * M : 2 * M;
    const float* mix = mixes + row * kMixRow;

    if (blockIdx.y == 0) {
        if (threadIdx.x >= kWave) return;
        const int l = lane();
        float c;
        if constexpr (COMB) {
            const int t = min(l, M * M - 1);
            const float logit = mix[2 * M + t] * scale[2] + base[2 * M + t];
            c = sinkhorn<M>(logit, hc_eps, sinkhorn_iters);
        } else {
            c = (l / M == l % M) ? 1.f : 0.f;
        }
        if (l < M)
            post_mix[row * M + l] =
                sigmoid(mix[M + l] * scale[1] + base[M + l]) * hc_post_alpha + (COMB ? 0.f : hc_eps);
        if (l < M * M) comb_mix[row * M * M + l] = c;
        return;
    }
    float pre[M];
#pragma unroll
    for (int i = 0; i < M; ++i) pre[i] = sigmoid(mix[i] * scale[0] + base[i]) + hc_eps;
    combine_streams<T, M, BLOCK, VEC>(pre, residual + row * M * H, layer_input + row * H, H,
                                      static_cast<int>(blockIdx.y) - 1);
}

// The streams after the layer: out_j = post[j]·x + Σ_i comb[i][j]·stream_i,
// block (n, c) over chunk c of row n. `residual` and `out` may be one plane:
// a thread reads its columns of every stream before it writes any.
template <class T, int M, int BLOCK, bool VEC>
__global__ __launch_bounds__(BLOCK) void hc_fold(const T* __restrict__ x, const T* residual,
                                                 const float* __restrict__ post_mix,
                                                 const float* __restrict__ comb_mix, T* out, int H,
                                                 const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const int chunk = static_cast<int>(blockIdx.y);
    float comb[M * M], post[M];
#pragma unroll
    for (int t = 0; t < M * M; ++t) comb[t] = comb_mix[row * M * M + t];
#pragma unroll
    for (int j = 0; j < M; ++j) post[j] = post_mix[row * M + j];
    const T* xr = x + row * H;
    const T* res = residual + row * M * H;
    T* dst = out + row * M * H;

    if constexpr (VEC) {
        const int h = column<BLOCK, VEC>(chunk, 0);
        if (h >= H) return;
        float xv[8], r[M][8];
        load8(xr + h, xv);
#pragma unroll
        for (int i = 0; i < M; ++i) load8(res + static_cast<long long>(i) * H + h, r[i]);
#pragma unroll
        for (int j = 0; j < M; ++j) {
            float acc[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                acc[k] = post[j] * xv[k];
#pragma unroll
                for (int i = 0; i < M; ++i) acc[k] += comb[i * M + j] * r[i][k];
            }
            store8(dst + static_cast<long long>(j) * H + h, acc);
        }
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int h = column<BLOCK, VEC>(chunk, k);
            if (h >= H) continue;
            const float xh = to_f32(xr[h]);
            float r[M];
#pragma unroll
            for (int i = 0; i < M; ++i) r[i] = to_f32(res[static_cast<long long>(i) * H + h]);
#pragma unroll
            for (int j = 0; j < M; ++j) {
                float acc = post[j] * xh;
#pragma unroll
                for (int i = 0; i < M; ++i) acc += comb[i * M + j] * r[i];
                dst[static_cast<long long>(j) * H + h] = from_f32<T>(acc);
            }
        }
    }
}

// The trunk's collapse before the head: y = Σ_i (sigmoid(logit_i) + eps)·stream_i,
// block (n, c) over chunk c of row n.
template <class T, int M, int BLOCK, bool VEC>
__global__ __launch_bounds__(BLOCK) void hc_head_postprocess(
    const float* __restrict__ mixes, const float* __restrict__ scale, const float* __restrict__ base,
    const T* __restrict__ residual, T* __restrict__ out, int H, float hc_eps,
    const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    float gates[M];
#pragma unroll
    for (int i = 0; i < M; ++i) gates[i] = sigmoid(mixes[row * M + i] * scale[0] + base[i]) + hc_eps;
    combine_streams<T, M, BLOCK, VEC>(gates, residual + row * M * H, out + row * H, H,
                                      static_cast<int>(blockIdx.y));
}

// Every stream of a row starts as the row: output[row][m][h] = input[row][h],
// block (n, c) over chunk c of row n.
template <class T, int M, int BLOCK, bool VEC>
__global__ __launch_bounds__(BLOCK) void hc_expand(const T* __restrict__ input,
                                                   T* __restrict__ output, int H,
                                                   const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const int chunk = static_cast<int>(blockIdx.y);
    const T* src = input + row * H;
    T* dst = output + row * M * H;
    if constexpr (VEC) {
        const int h = column<BLOCK, VEC>(chunk, 0);
        if (h >= H) return;
        const b128 v = *reinterpret_cast<const b128*>(src + h);
#pragma unroll
        for (int m = 0; m < M; ++m) *reinterpret_cast<b128*>(dst + static_cast<long long>(m) * H + h) = v;
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int h = column<BLOCK, VEC>(chunk, k);
            if (h >= H) continue;
            const T v = src[h];
#pragma unroll
            for (int m = 0; m < M; ++m) dst[static_cast<long long>(m) * H + h] = v;
        }
    }
}

// The stream row widened to f32 and scaled by rsqrt(mean(x²) + eps), no
// weight: the mix projection's input. One block a row; with VEC the row
// stays in registers between its sum and its store (at most CAP packs a
// thread, the launcher's check).
template <int BLOCK, int CAP, bool VEC>
__global__ __launch_bounds__(BLOCK) void hc_rmsnorm_f32(const bf16* __restrict__ input,
                                                        float* __restrict__ output, int dim,
                                                        float eps, const u32* __restrict__ win) {
    __shared__ float smem[BLOCK / kWave];
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const bf16* src = input + row * dim;
    float* dst = output + row * dim;
    if constexpr (VEC) {
        const int packs = dim / 8;
        float v[CAP][8];
        float local = 0.f;
#pragma unroll
        for (int c = 0; c < CAP; ++c) {
            const int p = c * BLOCK + static_cast<int>(threadIdx.x);
            if (p < packs) {
                load8(src + 8 * p, v[c]);
#pragma unroll
                for (int k = 0; k < 8; ++k) local += v[c][k] * v[c][k];
            }
        }
        const float s = rsqrtf(block_sum(local, smem) / static_cast<float>(dim) + eps);
#pragma unroll
        for (int c = 0; c < CAP; ++c) {
            const int p = c * BLOCK + static_cast<int>(threadIdx.x);
            if (p < packs) {
                float4* d = reinterpret_cast<float4*>(dst + 8 * p);
                d[0] = make_float4(v[c][0] * s, v[c][1] * s, v[c][2] * s, v[c][3] * s);
                d[1] = make_float4(v[c][4] * s, v[c][5] * s, v[c][6] * s, v[c][7] * s);
            }
        }
    } else {
        float local = 0.f;
        for (int d = threadIdx.x; d < dim; d += BLOCK) {
            const float v = to_f32(src[d]);
            local += v * v;
        }
        const float s = rsqrtf(block_sum(local, smem) / static_cast<float>(dim) + eps);
        for (int d = threadIdx.x; d < dim; d += BLOCK) dst[d] = to_f32(src[d]) * s;
    }
}

// The layer input from the streams: y[h] = mean_s(normed_s[h]·sigmoid(gate_s[h])),
// block (n, c) over chunk c of row n.
template <class T, int M, int BLOCK, bool VEC>
__global__ __launch_bounds__(BLOCK) void hc_mix(const T* __restrict__ gates,
                                                const T* __restrict__ normed, T* __restrict__ y,
                                                int H, const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const int chunk = static_cast<int>(blockIdx.y);
    const T* g = gates + row * M * H;
    const T* v = normed + row * M * H;
    T* dst = y + row * H;
    if constexpr (VEC) {
        const int h = column<BLOCK, VEC>(chunk, 0);
        if (h >= H) return;
        float gv[M][8], vv[M][8];
#pragma unroll
        for (int s = 0; s < M; ++s) {
            load8(g + static_cast<long long>(s) * H + h, gv[s]);
            load8(v + static_cast<long long>(s) * H + h, vv[s]);
        }
        float acc[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            acc[k] = 0.f;
#pragma unroll
            for (int s = 0; s < M; ++s) acc[k] += vv[s][k] / (1.f + __expf(-gv[s][k]));
            acc[k] /= static_cast<float>(M);
        }
        store8(dst + h, acc);
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int h = column<BLOCK, VEC>(chunk, k);
            if (h >= H) continue;
            float acc = 0.f;
#pragma unroll
            for (int s = 0; s < M; ++s) {
                const long long at = static_cast<long long>(s) * H + h;
                acc += to_f32(v[at]) / (1.f + __expf(-to_f32(g[at])));
            }
            dst[h] = from_f32<T>(acc / static_cast<float>(M));
        }
    }
}

// The layer's output back into every stream: hyper_s += 2·sigmoid(gate_s/M)·o,
// block (n, c) over chunk c of row n.
template <class T, int M, int BLOCK, bool VEC>
__global__ __launch_bounds__(BLOCK) void hc_inject(const T* __restrict__ o,
                                                   const T* __restrict__ gates,
                                                   T* __restrict__ hyper, int H,
                                                   const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const int chunk = static_cast<int>(blockIdx.y);
    const T* orow = o + row * H;
    T* hr = hyper + row * M * H;
    float g[M];
#pragma unroll
    for (int s = 0; s < M; ++s) g[s] = 2.f * fast_sigmoid(to_f32(gates[row * M + s]) / static_cast<float>(M));
    if constexpr (VEC) {
        const int h = column<BLOCK, VEC>(chunk, 0);
        if (h >= H) return;
        float ov[8], hv[M][8];
        load8(orow + h, ov);
#pragma unroll
        for (int s = 0; s < M; ++s) load8(hr + static_cast<long long>(s) * H + h, hv[s]);
#pragma unroll
        for (int s = 0; s < M; ++s) {
#pragma unroll
            for (int k = 0; k < 8; ++k) hv[s][k] = hv[s][k] + g[s] * ov[k];
            store8(hr + static_cast<long long>(s) * H + h, hv[s]);
        }
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int h = column<BLOCK, VEC>(chunk, k);
            if (h >= H) continue;
            const float oh = to_f32(orow[h]);
#pragma unroll
            for (int s = 0; s < M; ++s) {
                const long long at = static_cast<long long>(s) * H + h;
                hr[at] = from_f32<T>(to_f32(hr[at]) + g[s] * oh);
            }
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
