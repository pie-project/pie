#pragma once

// Row norms. One block of `BLOCK` threads (whole wavefronts) per normed row,
// f32 accumulation, one rounding on the store. A row whose width is a
// multiple of 8 and whose planes are 16-byte aligned moves in 16-byte packs.

#include "prelude/device.hip"

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

// y = rmsnorm(x) · w · gate(g), x and w in f32: the GDN output norm. The
// gate is silu(g), or sigmoid(g) when `sigmoid_gate` is set.
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
        yr[i] = from_f32<T>(w[i] * (xr[i] * inv) * gated);
    }
}

// y += x over the first `n` elements of a `width`-wide plane.
template <class T>
__global__ void residual_add(T* __restrict__ y, const T* __restrict__ x, usize n, int width,
                             const u32* __restrict__ win) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Window window{win};
    if (!window.live(static_cast<long long>(i / width))) return;
    const usize at = static_cast<usize>(window.plane(static_cast<long long>(i / width))) * width + i % width;
    y[at] = from_f32<T>(to_f32(y[at]) + to_f32(x[at]));
}

// x *= s, with s rounded to T first as the reference does.
template <class T>
__global__ void mul_scalar(T* __restrict__ x, float s, usize n, int width,
                           const u32* __restrict__ win) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Window window{win};
    if (!window.live(static_cast<long long>(i / width))) return;
    const usize at = static_cast<usize>(window.plane(static_cast<long long>(i / width))) * width + i % width;
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

}  // namespace pie::elemwise
