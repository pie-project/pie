#include "elemwise/row.cuh"

namespace pie::elemwise {

// A norm_gate row's tail, `u` its (scaled) values and `rv` its residual's:
// the norm, the gated sum (`gate_of(e, alpha, beta)` a channel's gate), `y`,
// and under NEXT the next norm into `z`.
template <class T, int BLOCK, bool NEXT, class Gate>
__device__ __forceinline__ void norm_gate_out(
    const float (&u)[8],
    const T (&rv)[8],
    bool gated,
    Gate gate_of,
    long long base,
    int at,
    T* __restrict__ y,
    float eps,
    const T* __restrict__ next_w,
    float next_eps,
    int plus_one,
    T* __restrict__ z,
    float* buf)
{
    constexpr int hidden = 8 * BLOCK;
    float local = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) local = fmaf(u[e], u[e], local);
    const float inv_rms = rsqrtf(block_reduce_sum_fast<BLOCK>(local, buf) / static_cast<float>(hidden) + eps);
    T out[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        if (!gated) {
            out[e] = Elem<T>::from_f32(u[e] * inv_rms);
        } else {
            const float n = Elem<T>::to_f32(Elem<T>::from_f32(u[e] * inv_rms));
            float alpha, beta;
            gate_of(e, alpha, beta);
            out[e] = Elem<T>::from_f32(alpha * Elem<T>::to_f32(rv[e]) + beta * n);
        }
    }
    *reinterpret_cast<uint4*>(y + base) = *reinterpret_cast<const uint4*>(out);
    if constexpr (NEXT) {
        T wt[8];
        *reinterpret_cast<uint4*>(wt) = *reinterpret_cast<const uint4*>(next_w + at);
        float sq = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float o = Elem<T>::to_f32(out[e]);
            sq += o * o;
        }
        __syncthreads();
        const float inv = rsqrtf(block_reduce_sum_fast<BLOCK>(sq, buf) / static_cast<float>(hidden) + next_eps);
        T zn[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            float wv = Elem<T>::to_f32(wt[e]);
            if (plus_one) wv += 1.f;
            zn[e] = Elem<T>::from_f32(Elem<T>::to_f32(out[e]) * inv * wv);
        }
        *reinterpret_cast<uint4*>(z + base) = *reinterpret_cast<const uint4*>(zn);
    }
}

// A gate's alpha and beta from its channel's `g`.
__device__ __forceinline__ void norm_gate_ab(float g, float temperature, float& alpha, float& beta)
{
    const float w = __fdividef(g, temperature);
    beta = __fdividef(1.f, 1.f + __expf(-w));
    const float a2 = fmaxf(__fdividef(1.f, 1.f + __expf(w)) * (1.f + beta), 1e-3f);
    alpha = a2 * rsqrtf(a2);
}

// `u = x * scale` rounded to T (`x` with no scale).
template <class T>
__device__ __forceinline__ void norm_gate_in(const T (&v)[8], const T* __restrict__ scale, int at, float (&u)[8])
{
    if (scale != nullptr) {
        T s[8];
        *reinterpret_cast<uint4*>(s) = *reinterpret_cast<const uint4*>(scale + at);
#pragma unroll
        for (int e = 0; e < 8; ++e)
            u[e] = Elem<T>::to_f32(Elem<T>::from_f32(Elem<T>::to_f32(v[e]) * Elem<T>::to_f32(s[e])));
    } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) u[e] = Elem<T>::to_f32(v[e]);
    }
}

// Muse Spark's post-norm and residual gate, a block a row of `8 * BLOCK`, a
// 16-byte run a thread held in registers: `u = x * scale` rounded to T (`x`
// with no scale), `n = u / rms(u)` rounded to T, and with a residual `y =
// alpha * res + beta * n` in f32 (`w = gate / temperature`, `beta =
// sigmoid(w)`, `alpha = sqrt(max(sigmoid(-w) * (1 + beta), 1e-3))`), else
// `y = n`. Under `NEXT` the next sublayer's input norm over `y` too: `z =
// rmsnorm(y) * w` (`1 + w` where `plus_one`).
template <class T, int BLOCK, bool NEXT>
__global__ void __launch_bounds__(BLOCK) norm_gate(
    const T* __restrict__ x,
    const T* __restrict__ scale,
    const T* __restrict__ res,
    const float* __restrict__ gate,
    T* __restrict__ y,
    float eps,
    float temperature,
    int fan,
    const T* __restrict__ next_w,
    float next_eps,
    int plus_one,
    T* __restrict__ z,
    const u32* __restrict__ win)
{
    constexpr int hidden = 8 * BLOCK;
    // A routed plane holds `fan` rows a token; the window counts tokens.
    const int row = blockIdx.x;
    if (win != nullptr && row >= static_cast<int>(win[0]) * fan) return;
    const int plane_row = win != nullptr ? row + static_cast<int>(win[1]) * fan : row;
    const int at = threadIdx.x * 8;
    const long long base = static_cast<long long>(plane_row) * hidden + at;
    T v[8];
    *reinterpret_cast<uint4*>(v) = *reinterpret_cast<const uint4*>(x + base);
    float u[8];
    norm_gate_in(v, scale, at, u);
    T rv[8];
    float g[8];
    if (res != nullptr) {
        *reinterpret_cast<uint4*>(rv) = *reinterpret_cast<const uint4*>(res + base);
        *reinterpret_cast<float4*>(g) = *reinterpret_cast<const float4*>(gate + at);
        *reinterpret_cast<float4*>(g + 4) = *reinterpret_cast<const float4*>(gate + at + 4);
    }
    __shared__ float buf[BLOCK];
    norm_gate_out<T, BLOCK, NEXT>(
        u, rv, res != nullptr, [&](int e, float& a, float& b) { norm_gate_ab(g[e], temperature, a, b); },
        base, at, y, eps, next_w, next_eps, plus_one, z, buf);
}

// norm_gate over a long fire, a block ROWS rows in turn: its channels' gate
// math taken once, and each row's loads out while the last is normed.
template <class T, int BLOCK, bool NEXT, int ROWS>
__global__ void __launch_bounds__(BLOCK) norm_gate_rows(
    const T* __restrict__ x,
    const T* __restrict__ scale,
    const T* __restrict__ res,
    const float* __restrict__ gate,
    T* __restrict__ y,
    float eps,
    float temperature,
    int fan,
    const T* __restrict__ next_w,
    float next_eps,
    int plus_one,
    T* __restrict__ z,
    int rows,
    const u32* __restrict__ win)
{
    constexpr int hidden = 8 * BLOCK;
    const int first = blockIdx.x * ROWS;
    const int end = win != nullptr ? min(rows, static_cast<int>(win[0]) * fan) : rows;
    const int last = min(first + ROWS, end);
    const long long shift = win != nullptr ? static_cast<long long>(win[1]) * fan : 0;
    const int at = threadIdx.x * 8;
    float alpha[8], beta[8];
    if (res != nullptr) {
        float g[8];
        *reinterpret_cast<float4*>(g) = *reinterpret_cast<const float4*>(gate + at);
        *reinterpret_cast<float4*>(g + 4) = *reinterpret_cast<const float4*>(gate + at + 4);
#pragma unroll
        for (int e = 0; e < 8; ++e) norm_gate_ab(g[e], temperature, alpha[e], beta[e]);
    }
    __shared__ float buf[BLOCK];
    uint4 xn{}, rn{};
    if (first < last) {
        xn = *reinterpret_cast<const uint4*>(x + (first + shift) * hidden + at);
        if (res != nullptr) rn = *reinterpret_cast<const uint4*>(res + (first + shift) * hidden + at);
    }
    for (int row = first; row < last; ++row) {
        const long long base = (row + shift) * hidden + at;
        T v[8], rv[8];
        *reinterpret_cast<uint4*>(v) = xn;
        *reinterpret_cast<uint4*>(rv) = rn;
        if (row + 1 < last) {
            xn = *reinterpret_cast<const uint4*>(x + base + hidden);
            if (res != nullptr) rn = *reinterpret_cast<const uint4*>(res + base + hidden);
        }
        float u[8];
        norm_gate_in(v, scale, at, u);
        if (row > first) __syncthreads();
        norm_gate_out<T, BLOCK, NEXT>(
            u, rv, res != nullptr, [&](int e, float& a, float& b) { a = alpha[e], b = beta[e]; },
            base, at, y, eps, next_w, next_eps, plus_one, z, buf);
    }
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

// The closed blocks a blend reads, each its own plane (`rows x H`), by value.
struct ResBlocks {
    const void* at[kMaxBlocks];
};

// A row's eight elements from 16 bytes, or its one element where the row is
// not read 16 bytes at a time.
template <class T>
__device__ __forceinline__ void res_load8(const T* v, i32 h, float (&x)[8]) {
    const uint4 w = *reinterpret_cast<const uint4*>(v + h);
    const T* e = reinterpret_cast<const T*>(&w);
#pragma unroll
    for (int i = 0; i < 8; ++i) x[i] = Elem<T>::to_f32(e[i]);
}

// AttnRes's blend of a token's closed blocks and its running prefix: each
// candidate's RMS-normed dot with the projection (one warp a candidate, its
// square sum and its dot in one pass), their softmax, and the blend of the
// candidates under it.
// With `kNorm` the blended row is RMS-normed under `out_weight` before it
// lands (`rmsnorm` of the blend's bf16 row, the two in one pass): its
// columns held in registers, at most `kTh * 8 * kChunks` of them, 16-byte
// runs only.
template <class T, int kTh, bool kNorm>
__device__ __forceinline__ void res_blend_body(
    const T* __restrict__ prefix,
    const ResBlocks& blocks,
    const T* __restrict__ norm_weight,
    const T* __restrict__ proj_weight,
    T* __restrict__ out,
    i32 B, i32 H, float eps,
    const T* __restrict__ out_weight, float out_eps,
    const u32* __restrict__ win)
{
    const i32 t = static_cast<i32>(blockIdx.x);

    if (win != nullptr && t >= static_cast<i32>(win[0])) return;

    const i32 row = win != nullptr ? t + static_cast<i32>(win[1]) : t;

    __shared__ float prob_s[kMaxBlocks + 1];

    const long long token_off = static_cast<long long>(row) * H;
    const i32 rows = B + 1;

    auto row_ptr = [&](i32 j) -> const T* {
        return (j < B) ? static_cast<const T*>(blocks.at[j]) + token_off : prefix + token_off;
    };
    // 16-byte runs where every row and both weights allow them.
    bool wide = (H % 8) == 0
        && (reinterpret_cast<unsigned long long>(norm_weight) & 15ull) == 0
        && (reinterpret_cast<unsigned long long>(proj_weight) & 15ull) == 0
        && (reinterpret_cast<unsigned long long>(out + token_off) & 15ull) == 0;
    for (i32 j = 0; j < rows; ++j) {
        wide = wide && (reinterpret_cast<unsigned long long>(row_ptr(j)) & 15ull) == 0;
    }

    // Each thread walks its columns of a group of candidates at once (their
    // loads independent of one another), its square sums and dots reduced
    // across the block a group at a time.
    constexpr int kGroup = 8;
    __shared__ float part[kTh / 32][2 * kGroup];
    const i32 lane = static_cast<i32>(threadIdx.x) & 31;
    const i32 warp = static_cast<i32>(threadIdx.x) >> 5;
    const i32 warps = static_cast<i32>(blockDim.x) >> 5;
    for (i32 j0 = 0; j0 < rows; j0 += kGroup) {
        float ss[kGroup];
        float dot[kGroup];
#pragma unroll
        for (int g = 0; g < kGroup; ++g) {
            ss[g] = 0.f;
            dot[g] = 0.f;
        }
        if (wide) {
            for (i32 h = static_cast<i32>(threadIdx.x) * 8; h < H;
                 h += static_cast<i32>(blockDim.x) * 8) {
                float nw[8], pw[8];
                res_load8(norm_weight, h, nw);
                res_load8(proj_weight, h, pw);
#pragma unroll
                for (int g = 0; g < kGroup; ++g) {
                    if (j0 + g < rows) {
                        float x[8];
                        res_load8(row_ptr(j0 + g), h, x);
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            ss[g] += x[i] * x[i];
                            dot[g] += x[i] * nw[i] * pw[i];
                        }
                    }
                }
            }
        } else {
            for (i32 h = static_cast<i32>(threadIdx.x); h < H;
                 h += static_cast<i32>(blockDim.x)) {
                const float w = Elem<T>::to_f32(norm_weight[h]) * Elem<T>::to_f32(proj_weight[h]);
#pragma unroll
                for (int g = 0; g < kGroup; ++g) {
                    if (j0 + g < rows) {
                        const float x = Elem<T>::to_f32(row_ptr(j0 + g)[h]);
                        ss[g] += x * x;
                        dot[g] += x * w;
                    }
                }
            }
        }
#pragma unroll
        for (int g = 0; g < kGroup; ++g) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                ss[g] += __shfl_xor_sync(0xffffffffu, ss[g], off);
                dot[g] += __shfl_xor_sync(0xffffffffu, dot[g], off);
            }
        }
        if (lane == 0) {
#pragma unroll
            for (int g = 0; g < kGroup; ++g) {
                part[warp][g] = ss[g];
                part[warp][kGroup + g] = dot[g];
            }
        }
        __syncthreads();
        if (threadIdx.x < kGroup && j0 + static_cast<i32>(threadIdx.x) < rows) {
            float s2 = 0.f;
            float d = 0.f;
            for (i32 w = 0; w < warps; ++w) {
                s2 += part[w][threadIdx.x];
                d += part[w][kGroup + threadIdx.x];
            }
            prob_s[j0 + threadIdx.x] = d * rsqrtf(s2 / static_cast<float>(H) + eps);
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

    if constexpr (kNorm) {
        // The blended row in registers, rounded to bf16 as it would land,
        // its square sum, then the norm.
        constexpr int kChunks = 4;
        float hv[kChunks][8];
        float ss = 0.f;
#pragma unroll
        for (int c = 0; c < kChunks; ++c) {
            const i32 h = (static_cast<i32>(threadIdx.x) + c * kTh) * 8;
#pragma unroll
            for (int i = 0; i < 8; ++i) hv[c][i] = 0.f;
            if (h < H) {
                for (i32 j = 0; j < rows; ++j) {
                    float x[8];
                    res_load8(row_ptr(j), h, x);
                    const float p = prob_s[j];
#pragma unroll
                    for (int i = 0; i < 8; ++i) hv[c][i] += p * x[i];
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    hv[c][i] = Elem<T>::to_f32(Elem<T>::from_f32(hv[c][i]));
                    ss += hv[c][i] * hv[c][i];
                }
            }
        }
        ss = ::pie::warp_sum(ss);
        if (lane == 0) part[warp][0] = ss;
        __syncthreads();
        float total = 0.f;
        for (i32 w = 0; w < warps; ++w) total += part[w][0];
        const float inv_rms = rsqrtf(total / static_cast<float>(H) + out_eps);
#pragma unroll
        for (int c = 0; c < kChunks; ++c) {
            const i32 h = (static_cast<i32>(threadIdx.x) + c * kTh) * 8;
            if (h < H) {
                float wv[8];
                res_load8(out_weight, h, wv);
                uint4 o;
                T* e = reinterpret_cast<T*>(&o);
#pragma unroll
                for (int i = 0; i < 8; ++i) e[i] = Elem<T>::from_f32(hv[c][i] * inv_rms * wv[i]);
                *reinterpret_cast<uint4*>(out + token_off + h) = o;
            }
        }
        return;
    }
    if (wide) {
        for (i32 h = static_cast<i32>(threadIdx.x) * 8; h < H;
             h += static_cast<i32>(blockDim.x) * 8) {
            float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            for (i32 j = 0; j < rows; ++j) {
                float x[8];
                res_load8(row_ptr(j), h, x);
                const float p = prob_s[j];
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[i] += p * x[i];
            }
            uint4 w;
            T* e = reinterpret_cast<T*>(&w);
#pragma unroll
            for (int i = 0; i < 8; ++i) e[i] = Elem<T>::from_f32(acc[i]);
            *reinterpret_cast<uint4*>(out + token_off + h) = w;
        }
    } else {
        for (i32 h = static_cast<i32>(threadIdx.x); h < H;
             h += static_cast<i32>(blockDim.x)) {
            float acc = 0.f;
            for (i32 j = 0; j < rows; ++j) {
                acc += prob_s[j] * Elem<T>::to_f32(row_ptr(j)[h]);
            }
            out[token_off + h] = Elem<T>::from_f32(acc);
        }
    }
}

template <class T>
__global__ void res_blend(
    const T* __restrict__ prefix,
    ResBlocks blocks,
    const T* __restrict__ norm_weight,
    const T* __restrict__ proj_weight,
    T* __restrict__ out,
    i32 B, i32 H, i32 block_rows, float eps,
    const u32* __restrict__ win)
{
    (void)block_rows;
    res_blend_body<T, kThreads, false>(
        prefix, blocks, norm_weight, proj_weight, out, B, H, eps, nullptr, 0.f, win);
}

// `res_blend` and the `rmsnorm` of its row (under `out_weight`, `out_eps`)
// in one pass, `kTh` threads a row; `out` the normed row.
template <class T, int kTh>
__global__ void __launch_bounds__(kTh) res_blend_norm(
    const T* __restrict__ prefix,
    ResBlocks blocks,
    const T* __restrict__ norm_weight,
    const T* __restrict__ proj_weight,
    const T* __restrict__ out_weight,
    T* __restrict__ out,
    i32 B, i32 H, float eps, float out_eps,
    const u32* __restrict__ win)
{
    res_blend_body<T, kTh, true>(
        prefix, blocks, norm_weight, proj_weight, out, B, H, eps, out_weight, out_eps, win);
}

constexpr int kModScaleShift = 0;
constexpr int kModScale = 1;
constexpr int kModTanhGate = 2;

__device__ __forceinline__ int modulation_row(
    const i32* __restrict__ lane_of_row, int row)
{
    return lane_of_row != nullptr ? lane_of_row[row] : row;
}

template <class T, class TM, int FORM>
__global__ void modulate(
    const T* __restrict__ x,
    const TM* __restrict__ m,
    const i32* __restrict__ lane_of_row,
    T* __restrict__ o,
    int width,
    int m_width,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const T* xr = x + static_cast<long long>(row) * width;
    T* orow = o + static_cast<long long>(row) * width;
    const TM* mr = m + static_cast<long long>(modulation_row(lane_of_row, row)) * m_width;

    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float xv = Elem<T>::to_f32(xr[i]);
        float v;
        if constexpr (FORM == kModScaleShift) {
            v = fmaf(xv, 1.f + Elem<TM>::to_f32(mr[i]), Elem<TM>::to_f32(mr[i + width]));
        } else if constexpr (FORM == kModScale) {
            v = xv * (1.f + Elem<TM>::to_f32(mr[i]));
        } else {
            v = tanhf(Elem<TM>::to_f32(mr[i])) * xv;
        }
        orow[i] = Elem<T>::from_f32(v);
    }
}

template <class T, class TM>
__global__ void gated_residual_add(
    const T* __restrict__ r,
    const TM* __restrict__ g,
    const T* __restrict__ y,
    const i32* __restrict__ lane_of_row,
    T* __restrict__ r_out,
    int width,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const T* rr = r + static_cast<long long>(row) * width;
    const T* yr = y + static_cast<long long>(row) * width;
    T* orow = r_out + static_cast<long long>(row) * width;
    const TM* gr = g + static_cast<long long>(modulation_row(lane_of_row, row)) * width;

    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        orow[i] = Elem<T>::from_f32(fmaf(Elem<TM>::to_f32(gr[i]),
                                         Elem<T>::to_f32(yr[i]),
                                         Elem<T>::to_f32(rr[i])));
    }
}

constexpr int kNormLayer = 0;
constexpr int kNormRmsNoScale = 1;
constexpr int kNormRmsWeight = 2;

template <class T, class TM, int BLOCK, int NORM, bool GATED>
__device__ __forceinline__ void norm_modulate_row(
    const T* __restrict__ src,
    const TM* __restrict__ g,
    const T* __restrict__ y,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    const TM* __restrict__ m,
    const i32* __restrict__ lane_of_row,
    T* __restrict__ o,
    int width,
    int m_width,
    float norm_eps,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int tid = threadIdx.x;
    const int mrow = modulation_row(lane_of_row, row);
    const TM* mr = m + static_cast<long long>(mrow) * m_width;
    T* orow = o + static_cast<long long>(row) * width;

    const T* normed = GATED ? residual + static_cast<long long>(row) * width
                            : src + static_cast<long long>(row) * width;

    __shared__ float buf[BLOCK];

    float local = 0.f;
    if constexpr (GATED) {
        const T* rr = src + static_cast<long long>(row) * width;
        const T* yr = y + static_cast<long long>(row) * width;
        const TM* gr = g + static_cast<long long>(mrow) * width;
        T* rout = residual + static_cast<long long>(row) * width;
        for (int i = tid; i < width; i += BLOCK) {
            const T summed = Elem<T>::from_f32(fmaf(Elem<TM>::to_f32(gr[i]),
                                                    Elem<T>::to_f32(yr[i]),
                                                    Elem<T>::to_f32(rr[i])));
            rout[i] = summed;
            const float v = Elem<T>::to_f32(summed);
            local += NORM == kNormLayer ? v : v * v;
        }
    } else {
        for (int i = tid; i < width; i += BLOCK) {
            const float v = Elem<T>::to_f32(normed[i]);
            local += NORM == kNormLayer ? v : v * v;
        }
    }

    float mean = 0.f;
    float inv = 0.f;
    if constexpr (NORM == kNormLayer) {
        mean = block_reduce_sum_fast<BLOCK>(local, buf) / static_cast<float>(width);
        __syncthreads();
        float spread = 0.f;
        for (int i = tid; i < width; i += BLOCK) {
            const float c = Elem<T>::to_f32(normed[i]) - mean;
            spread += c * c;
        }
        inv = rsqrtf(block_reduce_sum_fast<BLOCK>(spread, buf) /
                         static_cast<float>(width) +
                     norm_eps);
    } else {
        inv = rsqrtf(block_reduce_sum_fast<BLOCK>(local, buf) /
                         static_cast<float>(width) +
                     norm_eps);
    }

    for (int i = tid; i < width; i += BLOCK) {
        float c = (Elem<T>::to_f32(normed[i]) - mean) * inv;
        if constexpr (NORM == kNormRmsWeight) c *= Elem<T>::to_f32(weight[i]);
        orow[i] = Elem<T>::from_f32(
            fmaf(c, 1.f + Elem<TM>::to_f32(mr[i]), Elem<TM>::to_f32(mr[i + width])));
    }
}

template <class T, class TM, int BLOCK, int NORM>
__global__ void norm_modulate(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const TM* __restrict__ m,
    const i32* __restrict__ lane_of_row,
    T* __restrict__ o,
    int width,
    int m_width,
    float norm_eps,
    const u32* __restrict__ win)
{
    norm_modulate_row<T, TM, BLOCK, NORM, false>(
        x, nullptr, nullptr, nullptr, weight, m, lane_of_row, o, width, m_width,
        norm_eps, win);
}

template <class T, class TM, int BLOCK, int NORM>
__global__ void gated_residual_norm_modulate(
    const T* __restrict__ r,
    const TM* __restrict__ g,
    const T* __restrict__ y,
    T* __restrict__ r_out,
    const T* __restrict__ weight,
    const TM* __restrict__ m,
    const i32* __restrict__ lane_of_row,
    T* __restrict__ o,
    int width,
    int m_width,
    float norm_eps,
    const u32* __restrict__ win)
{
    norm_modulate_row<T, TM, BLOCK, NORM, true>(
        r, g, y, r_out, weight, m, lane_of_row, o, width, m_width, norm_eps, win);
}

}
