#pragma once

// Row norms. One block of `BLOCK` threads (whole wavefronts) per normed row,
// f32 accumulation, one rounding on the store. A row whose width is a
// multiple of 8 and whose planes are 16-byte aligned moves in 16-byte packs.

#include "prelude/device.hip"
#include "elemwise/pointwise.hip"

namespace pie::elemwise {

// Sum of squares of `n` elements of `x`, over the block.
template <class T, int BLOCK>
__device__ __forceinline__ float sum_squares(const T* __restrict__ x, int n, bool packed,
                                             float* smem) {
    float local = 0.f;
    if (packed) {
        for (int i = threadIdx.x; i < n / 8; i += BLOCK) {
            const Pack8<T> p = reinterpret_cast<const Pack8<T>*>(x)[i];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float v = to_f32(p.v[j]);
                local += v * v;
            }
        }
    } else {
        for (int i = threadIdx.x; i < n; i += BLOCK) {
            const float v = to_f32(x[i]);
            local += v * v;
        }
    }
    return block_sum(local, smem);
}

// y = x * rsqrt(mean(x²) + eps) * (w [+ 1]); `w` null scales by one.
template <class T, int BLOCK, bool PLUS_ONE>
__device__ __forceinline__ void norm_row(const T* __restrict__ x, const T* __restrict__ w,
                                         T* __restrict__ y, int n, float eps) {
    __shared__ float smem[BLOCK / kWave];
    const bool packed = n % 8 == 0 && aligned16(x) && aligned16(y) && (w == nullptr || aligned16(w));
    const float inv = rsqrtf(sum_squares<T, BLOCK>(x, n, packed, smem) / static_cast<float>(n) + eps);
    auto scale = [&](int i) {
        float s = w == nullptr ? 1.f : to_f32(w[i]);
        if constexpr (PLUS_ONE) s += 1.f;
        return s;
    };
    if (packed) {
        for (int i = threadIdx.x; i < n / 8; i += BLOCK) {
            const Pack8<T> p = reinterpret_cast<const Pack8<T>*>(x)[i];
            Pack8<T> q;
#pragma unroll
            for (int j = 0; j < 8; ++j) q.v[j] = from_f32<T>(to_f32(p.v[j]) * inv * scale(8 * i + j));
            reinterpret_cast<Pack8<T>*>(y)[i] = q;
        }
    } else {
        for (int i = threadIdx.x; i < n; i += BLOCK) y[i] = from_f32<T>(to_f32(x[i]) * inv * scale(i));
    }
}

// A launch over `rows * heads` blocks norms each `hidden`-wide head row;
// `heads` is 1 for a whole-row norm. Strides are in elements.
template <class T, int BLOCK, bool PLUS_ONE>
__device__ __forceinline__ void rmsnorm_rows(const T* x, const T* w, T* y, int hidden, int x_stride,
                                             int y_stride, float eps, int heads, const u32* win) {
    const Window window{win};
    if (!window.live(blockIdx.x, heads)) return;
    const long long row = window.plane(blockIdx.x, heads);
    norm_row<T, BLOCK, PLUS_ONE>(x + row * x_stride, w, y + row * y_stride, hidden, eps);
}

template <class T, int BLOCK>
__global__ __launch_bounds__(BLOCK) void rmsnorm(const T* __restrict__ x, const T* __restrict__ w,
                                                 T* __restrict__ y, int hidden, int x_stride,
                                                 int y_stride, float eps, int heads,
                                                 const u32* __restrict__ win) {
    rmsnorm_rows<T, BLOCK, false>(x, w, y, hidden, x_stride, y_stride, eps, heads, win);
}

template <class T, int BLOCK>
__global__ __launch_bounds__(BLOCK) void rmsnorm_plus_one(const T* __restrict__ x,
                                                          const T* __restrict__ w,
                                                          T* __restrict__ y, int hidden,
                                                          int x_stride, int y_stride, float eps,
                                                          int heads, const u32* __restrict__ win) {
    rmsnorm_rows<T, BLOCK, true>(x, w, y, hidden, x_stride, y_stride, eps, heads, win);
}

template <class T, int BLOCK>
__global__ __launch_bounds__(BLOCK) void rmsnorm_no_scale(const T* __restrict__ x,
                                                          T* __restrict__ y, int hidden, float eps,
                                                          int heads, const u32* __restrict__ win) {
    rmsnorm_rows<T, BLOCK, false>(x, nullptr, y, hidden, hidden, hidden, eps, heads, win);
}

// The per-head norms over contiguous heads of 8·LPH elements: LPH lanes a
// head row, eight elements a lane, the sum of squares across them by
// shuffles; 256 / LPH head rows a block, `total` (rows x heads) in all.
// `w` (one head wide) null scales by one. Planes 16-byte aligned.
template <class T, int LPH, bool PLUS_ONE>
__global__ __launch_bounds__(256) void rmsnorm_lanes(const T* __restrict__ x,
                                                     const T* __restrict__ w,
                                                     T* __restrict__ y, long long total,
                                                     float eps, int heads,
                                                     const u32* __restrict__ win) {
    constexpr int HD = 8 * LPH;
    const long long hr = static_cast<long long>(blockIdx.x) * (256 / LPH) + threadIdx.x / LPH;
    const Window window{win};
    if (hr >= total || !window.live(hr, heads)) return;
    const long long row = window.plane(hr, heads) * HD;
    const int c = (threadIdx.x % LPH) * 8;
    const Pack8<T> p = *reinterpret_cast<const Pack8<T>*>(x + row + c);
    float local = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float v = to_f32(p.v[e]);
        local += v * v;
    }
#pragma unroll
    for (int off = LPH / 2; off > 0; off >>= 1) local += __shfl_xor(local, off, kWave);
    const float inv = rsqrtf(local / static_cast<float>(HD) + eps);
    Pack8<T> ws;
    if (w != nullptr) ws = *reinterpret_cast<const Pack8<T>*>(w + c);
    Pack8<T> q;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        float sc = w == nullptr ? 1.f : to_f32(ws.v[e]);
        if constexpr (PLUS_ONE) sc += 1.f;
        q.v[e] = from_f32<T>(to_f32(p.v[e]) * inv * sc);
    }
    *reinterpret_cast<Pack8<T>*>(y + row + c) = q;
}

// y += x (the residual stream, rounded once), then out = rmsnorm(y) · (w [+ 1]).
template <class T, int BLOCK, bool PLUS_ONE>
__global__ __launch_bounds__(BLOCK) void residual_add_rmsnorm(const T* __restrict__ x,
                                                              T* __restrict__ y,
                                                              const T* __restrict__ w,
                                                              T* __restrict__ out, int hidden,
                                                              float eps,
                                                              const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x) * hidden;
    const T* xr = x + row;
    T* yr = y + row;
    const bool packed = hidden % 8 == 0 && aligned16(xr) && aligned16(yr);
    // A row of at most PPT packs a thread stays in registers from the add to
    // the scale: no store and reload of the stream between them. The sums
    // walk the same packs in the same order as `norm_row`'s.
    constexpr int PPT = 4;
    if (packed && hidden / 8 <= PPT * BLOCK && aligned16(out + row) && (w == nullptr || aligned16(w))) {
        __shared__ float smem[BLOCK / kWave];
        const int packs = hidden / 8;
        Pack8<T> b[PPT], wv[PPT];
#pragma unroll
        for (int p = 0; p < PPT; ++p) {
            const int i = threadIdx.x + p * BLOCK;
            if (i < packs) {
                const Pack8<T> a = reinterpret_cast<const Pack8<T>*>(xr)[i];
                b[p] = reinterpret_cast<const Pack8<T>*>(yr)[i];
                if (w != nullptr) wv[p] = reinterpret_cast<const Pack8<T>*>(w)[i];
#pragma unroll
                for (int j = 0; j < 8; ++j) b[p].v[j] = from_f32<T>(to_f32(b[p].v[j]) + to_f32(a.v[j]));
            }
        }
        float local = 0.f;
#pragma unroll
        for (int p = 0; p < PPT; ++p) {
            const int i = threadIdx.x + p * BLOCK;
            if (i < packs) {
                reinterpret_cast<Pack8<T>*>(yr)[i] = b[p];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float v = to_f32(b[p].v[j]);
                    local += v * v;
                }
            }
        }
        const float inv = rsqrtf(block_sum(local, smem) / static_cast<float>(hidden) + eps);
#pragma unroll
        for (int p = 0; p < PPT; ++p) {
            const int i = threadIdx.x + p * BLOCK;
            if (i < packs) {
                Pack8<T> q;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    float sc = w == nullptr ? 1.f : to_f32(wv[p].v[j]);
                    if constexpr (PLUS_ONE) sc += 1.f;
                    q.v[j] = from_f32<T>(to_f32(b[p].v[j]) * inv * sc);
                }
                reinterpret_cast<Pack8<T>*>(out + row)[i] = q;
            }
        }
        return;
    }
    if (packed) {
        for (int i = threadIdx.x; i < hidden / 8; i += BLOCK) {
            const Pack8<T> a = reinterpret_cast<const Pack8<T>*>(xr)[i];
            Pack8<T> b = reinterpret_cast<Pack8<T>*>(yr)[i];
#pragma unroll
            for (int j = 0; j < 8; ++j) b.v[j] = from_f32<T>(to_f32(b.v[j]) + to_f32(a.v[j]));
            reinterpret_cast<Pack8<T>*>(yr)[i] = b;
        }
    } else {
        for (int i = threadIdx.x; i < hidden; i += BLOCK) yr[i] = from_f32<T>(to_f32(yr[i]) + to_f32(xr[i]));
    }
    // The norm may walk the row in a different order than the add did.
    __syncthreads();
    norm_row<T, BLOCK, PLUS_ONE>(yr, w, out + row, hidden, eps);
}

// y = rmsnorm(x) · w · gate(g), x and w in f32 (w absent: scaleless): the
// GDN output norm. The gate is silu(g), or sigmoid(g) when `sigmoid_gate`
// is set.
template <class T, int BLOCK>
__global__ __launch_bounds__(BLOCK) void rmsnorm_gated_f32_in(const float* __restrict__ x,
                                                              const T* __restrict__ gate,
                                                              const float* __restrict__ w,
                                                              T* __restrict__ y, int hidden,
                                                              float eps, int sigmoid_gate,
                                                              int heads,
                                                              const u32* __restrict__ win) {
    __shared__ float smem[BLOCK / kWave];
    const Window window{win};
    if (!window.live(blockIdx.x, heads)) return;
    const long long row = window.plane(blockIdx.x, heads) * hidden;
    const float* xr = x + row;
    const T* gr = gate + row;
    T* yr = y + row;
    float local = 0.f;
    for (int i = threadIdx.x; i < hidden; i += BLOCK) local += xr[i] * xr[i];
    const float inv = rsqrtf(block_sum(local, smem) / static_cast<float>(hidden) + eps);
    for (int i = threadIdx.x; i < hidden; i += BLOCK) {
        const float g = to_f32(gr[i]);
        const float sig = 1.f / (1.f + __expf(-g));
        const float gated = sigmoid_gate ? sig : g * sig;
        yr[i] = from_f32<T>((w != nullptr ? w[i] : 1.f) * (xr[i] * inv) * gated);
    }
}

// rmsnorm_gated_f32_in over heads of 8·LPH elements: LPH lanes a head row,
// eight elements a lane, the sum of squares across them by shuffles; 256 /
// LPH head rows a block, `total` (rows x heads) in all. Planes 16-byte
// aligned.
template <class T, int LPH>
__global__ __launch_bounds__(256) void rmsnorm_gated_f32_lanes(const float* __restrict__ x,
                                                               const T* __restrict__ gate,
                                                               const float* __restrict__ w,
                                                               T* __restrict__ y, long long total,
                                                               float eps, int sigmoid_gate, int heads,
                                                               const u32* __restrict__ win) {
    constexpr int HD = 8 * LPH;
    const long long hr = static_cast<long long>(blockIdx.x) * (256 / LPH) + threadIdx.x / LPH;
    const Window window{win};
    if (hr >= total || !window.live(hr, heads)) return;
    const long long row = window.plane(hr, heads) * HD;
    const int c = (threadIdx.x % LPH) * 8;
    const float4 a = *reinterpret_cast<const float4*>(x + row + c);
    const float4 b = *reinterpret_cast<const float4*>(x + row + c + 4);
    const Pack8<T> g = *reinterpret_cast<const Pack8<T>*>(gate + row + c);
    const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    float ws[8] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    if (w != nullptr) {
        const float4 wa = *reinterpret_cast<const float4*>(w + c);
        const float4 wb = *reinterpret_cast<const float4*>(w + c + 4);
        const float wv[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) ws[e] = wv[e];
    }
    float local = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) local += v[e] * v[e];
#pragma unroll
    for (int off = LPH / 2; off > 0; off >>= 1) local += __shfl_xor(local, off, kWave);
    const float inv = rsqrtf(local / static_cast<float>(HD) + eps);
    Pack8<T> out;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float gv = to_f32(g.v[e]);
        const float sig = 1.f / (1.f + __expf(-gv));
        const float gated = sigmoid_gate ? sig : gv * sig;
        out.v[e] = from_f32<T>(ws[e] * (v[e] * inv) * gated);
    }
    *reinterpret_cast<Pack8<T>*>(y + row + c) = out;
}

// y += x over the first `n` elements of a `width`-wide plane: the live rows
// (the window's, else all) are one contiguous run, swept by a grid sized to
// the device rather than to the launch's padded extent, eight elements a
// thread where the rows allow.
template <class T>
__global__ void residual_add(T* __restrict__ y, const T* __restrict__ x, usize n, int width,
                             const u32* __restrict__ win) {
    const long long live = win == nullptr ? static_cast<long long>(n)
                                          : min(static_cast<long long>(n), static_cast<long long>(win[0]) * width);
    const long long base = win == nullptr ? 0 : static_cast<long long>(win[1]) * width;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    const long long first = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    T* yb = y + base;
    const T* xb = x + base;
    if (width % 8 == 0 && aligned16(yb) && aligned16(xb)) {
        for (long long p = first; p < live / 8; p += stride) {
            const Pack8<T> a = reinterpret_cast<const Pack8<T>*>(xb)[p];
            Pack8<T> b = reinterpret_cast<Pack8<T>*>(yb)[p];
#pragma unroll
            for (int e = 0; e < 8; ++e) b.v[e] = from_f32<T>(to_f32(b.v[e]) + to_f32(a.v[e]));
            reinterpret_cast<Pack8<T>*>(yb)[p] = b;
        }
        return;
    }
    for (long long i = first; i < live; i += stride) yb[i] = from_f32<T>(to_f32(yb[i]) + to_f32(xb[i]));
}

// x *= s, with s rounded to T first as the reference does.
template <class T>
__global__ void mul_scalar(T* __restrict__ x, float s, usize n, int width,
                           const u32* __restrict__ win) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long at = flat_at(win, static_cast<long long>(i), width);
    if (at < 0) return;
    x[at] = from_f32<T>(to_f32(x[at]) * to_f32(from_f32<T>(s)));
}

// out[n] += bias, one block per row.
template <class T, class TB>
__global__ void add_bias(T* __restrict__ out, const TB* __restrict__ bias, int dim,
                         const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    T* row = out + window.plane(blockIdx.x) * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) row[d] = from_f32<T>(to_f32(row[d]) + to_f32(bias[d]));
}

// x = silu(s · x).
template <class T>
__global__ void silu_scaled(T* __restrict__ x, float s, usize n, int width,
                            const u32* __restrict__ win) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long at = flat_at(win, static_cast<long long>(i), width);
    if (at < 0) return;
    const float v = to_f32(x[at]) * s;
    x[at] = from_f32<T>(v / (1.f + __expf(-v)));
}

// x *= s[0], the factor read from a one-element device plane.
template <class T>
__global__ void scale(T* __restrict__ x, const T* __restrict__ s, usize n, int width,
                      const u32* __restrict__ win) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long at = flat_at(win, static_cast<long long>(i), width);
    if (at < 0) return;
    x[at] = from_f32<T>(to_f32(x[at]) * to_f32(s[0]));
}

// out = (out - bias) · scale per column, one block per row. The launch is
// the plane's true extent: no window.
template <class T>
__global__ void standardize(T* __restrict__ out, const T* __restrict__ bias,
                            const T* __restrict__ scale, int dim) {
    T* row = out + static_cast<long long>(blockIdx.x) * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x)
        row[d] = from_f32<T>((to_f32(row[d]) - to_f32(bias[d])) * to_f32(scale[d]));
}

// y = rmsnorm(x) · w (`w + 1` under PLUS_ONE) over each `group`-wide slice
// of a row; the weight spans the full row, one slice of it per group. A
// launch over `rows * groups` blocks.
template <class T, int BLOCK, bool PLUS_ONE>
__global__ __launch_bounds__(BLOCK) void rmsnorm_grouped(const T* __restrict__ x, const T* __restrict__ w,
                                                         T* __restrict__ y, int group, int groups, float eps,
                                                         const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x, groups)) return;
    const long long at = window.plane(blockIdx.x, groups) * group;
    const T* wg = w + static_cast<long long>(blockIdx.x % groups) * group;
    norm_row<T, BLOCK, PLUS_ONE>(x + at, wg, y + at, group, eps);
}

// y = rmsnorm(o) · w · sigmoid(g) per `D`-wide head, o and w in f32: the
// KDA output norm. Grid (rows, H); a block of one or two wavefronts walks
// one head.
template <class T>
__global__ __launch_bounds__(128) void rmsnorm_gated_by(const float* __restrict__ o,
                                                        const T* __restrict__ g,
                                                        const float* __restrict__ w,
                                                        T* __restrict__ y, int H, int D, float eps,
                                                        const u32* __restrict__ win) {
    __shared__ float smem[128 / kWave];
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long base = (window.plane(blockIdx.x) * H + blockIdx.y) * D;
    float local = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) local += o[base + d] * o[base + d];
    const float inv = rsqrtf(block_sum(local, smem) / static_cast<float>(D) + eps);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float gate = 1.f / (1.f + __expf(-to_f32(g[base + d])));
        y[base + d] = from_f32<T>(o[base + d] * inv * w[d] * gate);
    }
}

// Muse Spark's post-norm and residual gate, a block a `n`-wide row: `u = x *
// scale` rounded to T, `m = u / rms(u)` rounded, and with a residual `y =
// alpha * res + beta * m` (`w = gate / temperature`, `beta = sigmoid(w)`,
// `alpha = sqrt(max(sigmoid(-w) * (1 + beta), 1e-3))`), else `y = m`. Under
// `NEXT` also `z = rmsnorm(y) * (next_w [+ 1])`. A routed plane holds `fan`
// rows a token.
template <class T, int BLOCK, bool NEXT>
__global__ __launch_bounds__(BLOCK) void norm_gate(
    const T* __restrict__ x, const T* __restrict__ scale, const T* res,
    const float* __restrict__ gate, T* y, int n, float eps, float temperature, int fan,
    const T* __restrict__ next_w, float next_eps, int plus_one, T* __restrict__ z,
    const u32* __restrict__ win) {
    __shared__ float smem[BLOCK / kWave];
    const Window window{win};
    if (!window.live(blockIdx.x, fan)) return;
    const long long base = window.plane(blockIdx.x, fan) * n;
    auto scaled = [&](int i) {
        const float v = to_f32(x[base + i]);
        return scale == nullptr ? v : to_f32(from_f32<T>(v * to_f32(scale[i])));
    };
    float local = 0.f;
    for (int i = threadIdx.x; i < n; i += BLOCK) {
        const float u = scaled(i);
        local = fmaf(u, u, local);
    }
    const float inv = rsqrtf(block_sum(local, smem) / static_cast<float>(n) + eps);
    float sq = 0.f;
    for (int i = threadIdx.x; i < n; i += BLOCK) {
        float o = scaled(i) * inv;
        if (res != nullptr) {
            const float w = gate[i] / temperature;
            const float beta = 1.f / (1.f + __expf(-w));
            const float alpha = sqrtf(fmaxf((1.f / (1.f + __expf(w))) * (1.f + beta), 1e-3f));
            o = alpha * to_f32(res[base + i]) + beta * to_f32(from_f32<T>(o));
        }
        const T out = from_f32<T>(o);
        y[base + i] = out;
        sq += to_f32(out) * to_f32(out);
    }
    if constexpr (NEXT) {
        const float inv2 = rsqrtf(block_sum(sq, smem) / static_cast<float>(n) + next_eps);
        for (int i = threadIdx.x; i < n; i += BLOCK) {
            const float w = to_f32(next_w[i]) + (plus_one ? 1.f : 0.f);
            z[base + i] = from_f32<T>(to_f32(y[base + i]) * inv2 * w);
        }
    }
}

// norm_gate over a row of whole eights in 16-byte aligned planes: the row's
// packs held in registers, CH a thread, through both norms.
template <class T, int BLOCK, bool NEXT, int CH>
__global__ __launch_bounds__(BLOCK) void norm_gate_vec8(
    const T* __restrict__ x, const T* __restrict__ scale, const T* res,
    const float* __restrict__ gate, T* y, int n, float eps, float temperature, int fan,
    const T* __restrict__ next_w, float next_eps, int plus_one, T* __restrict__ z,
    const u32* __restrict__ win) {
    __shared__ float smem[BLOCK / kWave];
    const Window window{win};
    if (!window.live(blockIdx.x, fan)) return;
    const long long base = window.plane(blockIdx.x, fan) * n;
    const int packs = n / 8, tid = threadIdx.x;
    float u[CH][8];
    float local = 0.f;
#pragma unroll
    for (int c = 0; c < CH; ++c) {
        const int k = min(tid + c * BLOCK, packs - 1);
        const Pack8<T> xv = reinterpret_cast<const Pack8<T>*>(x + base)[k];
        Pack8<T> sv;
        if (scale != nullptr) sv = reinterpret_cast<const Pack8<T>*>(scale)[k];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float v = to_f32(xv.v[e]);
            u[c][e] = scale == nullptr ? v : to_f32(from_f32<T>(v * to_f32(sv.v[e])));
            if (tid + c * BLOCK < packs) local = fmaf(u[c][e], u[c][e], local);
        }
    }
    const float inv = rsqrtf(block_sum(local, smem) / static_cast<float>(n) + eps);
    float sq = 0.f;
    Pack8<T> outs[CH];
#pragma unroll
    for (int c = 0; c < CH; ++c) {
        const int k = tid + c * BLOCK;
        if (k >= packs) continue;
        Pack8<T> rv;
        float gv[8];
        if (res != nullptr) {
            rv = reinterpret_cast<const Pack8<T>*>(res + base)[k];
            const float4 ga = reinterpret_cast<const float4*>(gate)[2 * k];
            const float4 gb = reinterpret_cast<const float4*>(gate)[2 * k + 1];
            const float g8[8] = {ga.x, ga.y, ga.z, ga.w, gb.x, gb.y, gb.z, gb.w};
#pragma unroll
            for (int e = 0; e < 8; ++e) gv[e] = g8[e];
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            float o = u[c][e] * inv;
            if (res != nullptr) {
                const float w = gv[e] / temperature;
                const float beta = 1.f / (1.f + __expf(-w));
                const float alpha = sqrtf(fmaxf((1.f / (1.f + __expf(w))) * (1.f + beta), 1e-3f));
                o = alpha * to_f32(rv.v[e]) + beta * to_f32(from_f32<T>(o));
            }
            outs[c].v[e] = from_f32<T>(o);
            sq += to_f32(outs[c].v[e]) * to_f32(outs[c].v[e]);
        }
        reinterpret_cast<Pack8<T>*>(y + base)[k] = outs[c];
    }
    if constexpr (NEXT) {
        const float inv2 = rsqrtf(block_sum(sq, smem) / static_cast<float>(n) + next_eps);
#pragma unroll
        for (int c = 0; c < CH; ++c) {
            const int k = tid + c * BLOCK;
            if (k >= packs) continue;
            const Pack8<T> wv = reinterpret_cast<const Pack8<T>*>(next_w)[k];
            Pack8<T> q;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float w = to_f32(wv.v[e]) + (plus_one ? 1.f : 0.f);
                q.v[e] = from_f32<T>(to_f32(outs[c].v[e]) * inv2 * w);
            }
            reinterpret_cast<Pack8<T>*>(z + base)[k] = q;
        }
    }
}

// `V` elements moved as one access: a 16-byte pack of eight, or one.
template <class T, int V>
struct alignas(sizeof(T) * V) Lanes {
    T v[V];
};

// The fused norm-then-residual chain of a sandwich-norm block:
//   t = rmsnorm(x) · w0, y += t, [scaled = y · s[0]], [out = rmsnorm(last) · (w1 [+ 1])]
// where `last` is the scaled stream when there is one, else the stream.
// Every step rounds to T as it is stored and the next reads the rounded
// value. A row of at most CHUNKS · BLOCK groups of V elements stays in
// registers through the chain; loads index a clamped group so they issue
// unconditionally, and the tail groups are masked out of the sums.
template <class T, int BLOCK, int V, int CHUNKS, bool SCALE, bool POST, bool POST_PLUS_ONE>
__device__ __forceinline__ void residual_chain(const T* __restrict__ x, const T* __restrict__ w0,
                                               T* __restrict__ t, T* __restrict__ y,
                                               const T* __restrict__ s, T* __restrict__ scaled,
                                               const T* __restrict__ w1, T* __restrict__ out,
                                               int hidden, float eps0, float eps1,
                                               const u32* __restrict__ win) {
    using L = Lanes<T, V>;
    __shared__ float smem[BLOCK / kWave];
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long base = window.plane(blockIdx.x) * hidden;
    const int groups = hidden / V;
    const int tid = static_cast<int>(threadIdx.x);

    L xv[CHUNKS], yv[CHUNKS], wv0[CHUNKS], wv1[CHUNKS];
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int k = min(tid + c * BLOCK, groups - 1);
        xv[c] = reinterpret_cast<const L*>(x + base)[k];
        yv[c] = reinterpret_cast<const L*>(y + base)[k];
        wv0[c] = reinterpret_cast<const L*>(w0)[k];
        if constexpr (POST) wv1[c] = reinterpret_cast<const L*>(w1)[k];
    }
    float local = 0.f;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        if (tid + c * BLOCK < groups) {
#pragma unroll
            for (int j = 0; j < V; ++j) {
                const float v = to_f32(xv[c].v[j]);
                local += v * v;
            }
        }
    }
    const float inv0 = rsqrtf(block_sum(local, smem) / static_cast<float>(hidden) + eps0);
    const float sf = SCALE ? to_f32(s[0]) : 1.f;

    L last[CHUNKS];
    float local1 = 0.f;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int i = tid + c * BLOCK;
        if (i < groups) {
            L tq, yq;
#pragma unroll
            for (int j = 0; j < V; ++j) {
                tq.v[j] = from_f32<T>(to_f32(xv[c].v[j]) * inv0 * to_f32(wv0[c].v[j]));
                yq.v[j] = from_f32<T>(to_f32(yv[c].v[j]) + to_f32(tq.v[j]));
            }
            reinterpret_cast<L*>(t + base)[i] = tq;
            reinterpret_cast<L*>(y + base)[i] = yq;
            last[c] = yq;
            if constexpr (SCALE) {
#pragma unroll
                for (int j = 0; j < V; ++j) last[c].v[j] = from_f32<T>(to_f32(yq.v[j]) * sf);
                reinterpret_cast<L*>(scaled + base)[i] = last[c];
            }
#pragma unroll
            for (int j = 0; j < V; ++j) {
                const float v = to_f32(last[c].v[j]);
                local1 += v * v;
            }
        }
    }
    if constexpr (POST) {
        const float inv1 = rsqrtf(block_sum(local1, smem) / static_cast<float>(hidden) + eps1);
#pragma unroll
        for (int c = 0; c < CHUNKS; ++c) {
            const int i = tid + c * BLOCK;
            if (i < groups) {
                L oq;
#pragma unroll
                for (int j = 0; j < V; ++j) {
                    float sc = to_f32(wv1[c].v[j]);
                    if constexpr (POST_PLUS_ONE) sc += 1.f;
                    oq.v[j] = from_f32<T>(to_f32(last[c].v[j]) * inv1 * sc);
                }
                reinterpret_cast<L*>(out + base)[i] = oq;
            }
        }
    }
}

// The chain element by element: any T, any width up to PER_THREAD · BLOCK.
template <class T, int BLOCK, int PER_THREAD, bool SCALE, bool POST, bool POST_PLUS_ONE>
__global__ __launch_bounds__(BLOCK) void rmsnorm_residual_add(
    const T* __restrict__ x, const T* __restrict__ w0, T* __restrict__ t, T* __restrict__ y,
    const T* __restrict__ s, T* __restrict__ scaled, const T* __restrict__ w1, T* __restrict__ out,
    int hidden, float eps0, float eps1, const u32* __restrict__ win) {
    residual_chain<T, BLOCK, 1, PER_THREAD, SCALE, POST, POST_PLUS_ONE>(
        x, w0, t, y, s, scaled, w1, out, hidden, eps0, eps1, win);
}

// The chain in 16-byte packs: bf16, a width that is a multiple of 8, every
// plane 16-byte aligned.
template <int BLOCK, int CHUNKS, bool SCALE, bool POST, bool POST_PLUS_ONE>
__global__ __launch_bounds__(BLOCK) void rmsnorm_residual_add_vec8(
    const bf16* __restrict__ x, const bf16* __restrict__ w0, bf16* __restrict__ t,
    bf16* __restrict__ y, const bf16* __restrict__ s, bf16* __restrict__ scaled,
    const bf16* __restrict__ w1, bf16* __restrict__ out, int hidden, float eps0, float eps1,
    const u32* __restrict__ win) {
    residual_chain<bf16, BLOCK, 8, CHUNKS, SCALE, POST, POST_PLUS_ONE>(
        x, w0, t, y, s, scaled, w1, out, hidden, eps0, eps1, win);
}

// The candidates a residual blend may weigh, besides the prefix.
constexpr int kBlendMax = 32;

// The attention-residual blend: a row's candidates are the B closed blocks'
// rows (stacked [B][block_rows][H]) and the prefix row. Each scores
// <rmsnorm(v) · norm_w, proj_w>; a softmax over the B + 1 scores weighs them,
// and out is their weighted sum. One block of 256 threads per row.
template <class T>
__global__ __launch_bounds__(256) void res_blend(const T* __restrict__ prefix,
                                                 const T* __restrict__ blocks,
                                                 const T* __restrict__ norm_w,
                                                 const T* __restrict__ proj_w, T* __restrict__ out,
                                                 i32 B, i32 H, i32 block_rows, float eps,
                                                 const u32* __restrict__ win) {
    constexpr int kWaves = 256 / kWave;
    __shared__ float part[2][kWaves];
    __shared__ float prob[kBlendMax + 1];
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const long long at = row * H;
    auto candidate = [&](int j) {
        return j < B ? blocks + (static_cast<long long>(j) * block_rows + row) * H : prefix + at;
    };
    const int w = static_cast<int>(threadIdx.x) / kWave;

    // A score is rsqrt(mean(v²) + eps) · Σ v · norm_w · proj_w: the sum of
    // squares and the projection share one pass over the candidate.
    for (int j = 0; j <= B; ++j) {
        const T* v = candidate(j);
        float ss = 0.f, dot = 0.f;
        for (int h = threadIdx.x; h < H; h += 256) {
            const float x = to_f32(v[h]);
            ss += x * x;
            dot += x * to_f32(norm_w[h]) * to_f32(proj_w[h]);
        }
        ss = wave_sum(ss);
        dot = wave_sum(dot);
        if (lane() == 0) {
            part[0][w] = ss;
            part[1][w] = dot;
        }
        lds_barrier();
        if (threadIdx.x == 0) {
            float s2 = 0.f, d = 0.f;
            for (int i = 0; i < kWaves; ++i) {
                s2 += part[0][i];
                d += part[1][i];
            }
            prob[j] = d * rsqrtf(s2 / static_cast<float>(H) + eps);
        }
        lds_barrier();
    }

    if (threadIdx.x == 0) {
        float m = prob[0];
        for (int j = 1; j <= B; ++j) m = fmaxf(m, prob[j]);
        float sum = 0.f;
        for (int j = 0; j <= B; ++j) {
            prob[j] = __expf(prob[j] - m);
            sum += prob[j];
        }
        const float inv = 1.f / sum;
        for (int j = 0; j <= B; ++j) prob[j] *= inv;
    }
    lds_barrier();

    for (int h = threadIdx.x; h < H; h += 256) {
        float acc = 0.f;
        for (int j = 0; j <= B; ++j) acc += prob[j] * to_f32(candidate(j)[h]);
        out[at + h] = from_f32<T>(acc);
    }
}

}  // namespace pie::elemwise
