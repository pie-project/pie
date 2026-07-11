#pragma once

// PTIR tier-0 op-kernel library (docs/ptir/thrust-3-programs.md P4.2, overview
// §7.3 "Tier 0 — interpret"). One prebuilt row-parallel CUDA kernel per core op
// family (overview appendix). The stage-runner walks a validated trace launch by
// launch, dispatching each op to the matching kernel here; `top_k` / `matmul`
// link as library kernels (§7.3), never generated.
//
// Layout contract: every tensor is a row-major device buffer. An op over a
// `[rows, len]` shape treats the leading dim as the row (CTA) axis and the
// trailing dims flattened as the per-row length — row-local reductions/scans map
// one CTA (256 threads) to one row; element-wise maps grid-stride over numel;
// gather/scatter grid-stride over the index/source length. This is the entire
// Metal porting surface (§7.3): a few dozen row-parallel kernels.
//
// Compiled by nvcc (part of the driver lib / the tier-0 test). Tier 1 (P5)
// re-emits the same math fused per stage via NVRTC (sampling_ir codegen
// generalized); this library is the correctness baseline every tier diffs
// against echo's host golden interpreter.
//
// RNG PARITY: the gumbel/uniform helpers are transcribed byte-for-byte from
// sampling_ir/primitives_src.hpp (SplitMix64, seed_eff ^ 0xA5A5A5A5, stream
// salt) so tier-0 sampling matches the shipped sample_temp path. Do not
// "improve" the constants.

#include <cstdint>

#include <cuda_runtime.h>
#include <math_constants.h>

#include "pie_native/ptir/op_table.hpp"

namespace pie_cuda_driver::ptir {

inline constexpr int kTier0Block = 256;   // fixed CTA width for row-local kernels

// ───────────────────────────── device helpers ────────────────────────────

__device__ __forceinline__ float t0_pos_inf() { return __int_as_float(0x7f800000); }
__device__ __forceinline__ float t0_neg_inf() { return __int_as_float(0xff800000); }

// RNG — bit-parity with sampling_ir/primitives_src.hpp (sample_temp.cu lineage).
__device__ __forceinline__ unsigned long long t0_splitmix64(unsigned long long x) {
    x ^= x >> 27; x *= 0x3C79AC492BA7B653ULL;
    x ^= x >> 33; x *= 0x1C69B3F74AC4AE35ULL;
    x ^= x >> 27;
    return x;
}
__device__ __forceinline__ unsigned long long t0_seed_eff(unsigned int seed_u32) {
    return (unsigned long long)seed_u32 ^ 0xA5A5A5A5ULL;
}
__device__ __forceinline__ unsigned long long t0_stream_salt(unsigned int stream) {
    return t0_splitmix64((unsigned long long)stream * 0x9E3779B97F4A7C15ULL);
}
__device__ __forceinline__ unsigned long long t0_seed_eff_stream(unsigned int seed_u32,
                                                                 unsigned int stream) {
    return t0_seed_eff(seed_u32) ^ t0_stream_salt(stream);
}
__device__ __forceinline__ float t0_hash_uniform(unsigned long long seed_eff, int j) {
    unsigned long long x = seed_eff + 0x9E3779B97F4A7C15ULL * (unsigned long long)(j + 1);
    x = t0_splitmix64(x);
    unsigned int bits = (unsigned int)(x >> 40);
    return ((float)bits + 0.5f) * (1.0f / 16777216.0f);
}
__device__ __forceinline__ float t0_gumbel_noise(unsigned long long seed_eff, int j) {
    float u = t0_hash_uniform(seed_eff, j);
    return -logf(-logf(u));
}

// ─────────────────────────── map / element-wise ──────────────────────────

enum class BinKind : std::uint8_t { Add, Sub, Mul, Div, Rem, MaxElem, MinElem };

template <class T>
__device__ __forceinline__ T t0_bin(BinKind k, T a, T b) {
    switch (k) {
        case BinKind::Add: return a + b;
        case BinKind::Sub: return a - b;
        case BinKind::Mul: return a * b;
        case BinKind::Div: return a / b;
        case BinKind::Rem: return a - (a / b) * b;   // integer-style remainder
        case BinKind::MaxElem: return a > b ? a : b;
        case BinKind::MinElem: return a < b ? a : b;
    }
    return a;
}
template <>
__device__ __forceinline__ float t0_bin<float>(BinKind k, float a, float b) {
    switch (k) {
        case BinKind::Add: return a + b;
        case BinKind::Sub: return a - b;
        case BinKind::Mul: return a * b;
        case BinKind::Div: return a / b;
        case BinKind::Rem: return fmodf(a, b);
        case BinKind::MaxElem: return fmaxf(a, b);
        case BinKind::MinElem: return fminf(a, b);
    }
    return a;
}

template <class T>
__global__ void k_binary(const T* __restrict__ a, const T* __restrict__ b,
                         T* __restrict__ out, std::uint64_t n, BinKind kind,
                         int a_scalar = 0, int b_scalar = 0) {
    for (std::uint64_t i = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x;
         i < n; i += (std::uint64_t)gridDim.x * blockDim.x) {
        out[i] = t0_bin<T>(kind, a[a_scalar ? 0 : i], b[b_scalar ? 0 : i]);
    }
}

enum class UnKind : std::uint8_t { Neg, Exp, Log, Recip, Abs, Sign };

template <class T>
__global__ void k_unary(const T* __restrict__ a, T* __restrict__ out,
                        std::uint64_t n, UnKind kind) {
    for (std::uint64_t i = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x;
         i < n; i += (std::uint64_t)gridDim.x * blockDim.x) {
        T v = a[i];
        switch (kind) {
            case UnKind::Neg:   out[i] = -v; break;
            case UnKind::Exp:   out[i] = (T)expf((float)v); break;
            case UnKind::Log:   out[i] = (T)logf((float)v); break;
            case UnKind::Recip: out[i] = (T)(1.0f / (float)v); break;
            case UnKind::Abs:   out[i] = (T)fabsf((float)v); break;
            case UnKind::Sign:  out[i] = (T)(((float)v > 0.0f) - ((float)v < 0.0f)); break;
        }
    }
}

enum class CmpKind : std::uint8_t { Eq, Ne, Lt, Le, Gt, Ge };

template <class T>
__global__ void k_compare(const T* __restrict__ a, const T* __restrict__ b,
                          std::uint8_t* __restrict__ out, std::uint64_t n, CmpKind kind,
                          int a_scalar = 0, int b_scalar = 0) {
    for (std::uint64_t i = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x;
         i < n; i += (std::uint64_t)gridDim.x * blockDim.x) {
        T x = a[a_scalar ? 0 : i], y = b[b_scalar ? 0 : i];
        bool r = false;
        switch (kind) {
            case CmpKind::Eq: r = (x == y); break;
            case CmpKind::Ne: r = (x != y); break;
            case CmpKind::Lt: r = (x <  y); break;
            case CmpKind::Le: r = (x <= y); break;
            case CmpKind::Gt: r = (x >  y); break;
            case CmpKind::Ge: r = (x >= y); break;
        }
        out[i] = r ? 1u : 0u;
    }
}

enum class LogicKind : std::uint8_t { And, Or };

__global__ void k_logic(const std::uint8_t* __restrict__ a, const std::uint8_t* __restrict__ b,
                        std::uint8_t* __restrict__ out, std::uint64_t n, LogicKind kind) {
    for (std::uint64_t i = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x;
         i < n; i += (std::uint64_t)gridDim.x * blockDim.x) {
        std::uint8_t x = a[i] ? 1u : 0u, y = b[i] ? 1u : 0u;
        out[i] = (kind == LogicKind::And) ? (x & y) : (x | y);
    }
}

__global__ void k_not(const std::uint8_t* __restrict__ a, std::uint8_t* __restrict__ out,
                      std::uint64_t n) {
    for (std::uint64_t i = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x;
         i < n; i += (std::uint64_t)gridDim.x * blockDim.x) {
        out[i] = a[i] ? 0u : 1u;
    }
}

// select(cond, a, b): cond != 0 → a else b (§2, the data-dependent branch).
template <class T>
__global__ void k_select(const std::uint8_t* __restrict__ cond, const T* __restrict__ a,
                         const T* __restrict__ b, T* __restrict__ out, std::uint64_t n,
                         int a_scalar = 0, int b_scalar = 0) {
    for (std::uint64_t i = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x;
         i < n; i += (std::uint64_t)gridDim.x * blockDim.x) {
        out[i] = cond[i] ? a[a_scalar ? 0 : i] : b[b_scalar ? 0 : i];
    }
}

// cast — dtype conversion (bool source treated as {0,1}).
template <class TIn, class TOut>
__global__ void k_cast(const TIn* __restrict__ in, TOut* __restrict__ out, std::uint64_t n) {
    for (std::uint64_t i = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x;
         i < n; i += (std::uint64_t)gridDim.x * blockDim.x) {
        out[i] = (TOut)in[i];
    }
}

// ───────────────────────────────── index ─────────────────────────────────

__global__ void k_iota(std::uint32_t* __restrict__ out, std::uint32_t n) {
    for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        out[i] = i;
    }
}

// gather: out[i] = src[idx[i]]  (idx length = output length).
template <class T>
__global__ void k_gather(const T* __restrict__ src, const std::uint32_t* __restrict__ idx,
                         T* __restrict__ out, std::uint64_t n_idx) {
    for (std::uint64_t i = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x;
         i < n_idx; i += (std::uint64_t)gridDim.x * blockDim.x) {
        out[i] = src[idx[i]];
    }
}

// gather_row (axis-0 row gather): src [n, row_len], idx [m] → out [m, row_len],
// out[i, :] = src[idx[i], :]. Grid-stride over m*row_len.
template <class T>
__global__ void k_gather_row(const T* __restrict__ src, const std::uint32_t* __restrict__ idx,
                             T* __restrict__ out, std::uint32_t m, std::uint32_t row_len) {
    std::uint64_t n = (std::uint64_t)m * row_len;
    for (std::uint64_t t = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x;
         t < n; t += (std::uint64_t)gridDim.x * blockDim.x) {
        std::uint32_t i = (std::uint32_t)(t / row_len);
        std::uint32_t j = (std::uint32_t)(t % row_len);
        out[t] = src[(std::uint64_t)idx[i] * row_len + j];
    }
}

// scatter_set: out starts as a copy of base; out[idx[j]] = vals[j] for j in
// index order, LAST WINS on duplicates (§6.2, load-bearing). Tier-0 correctness
// path: a single ordered pass (scatter counts are small — K/B words per lane).
// Tier 1 fuses this into a few flat writes per lane.
template <class T>
__global__ void k_scatter_set_serial(T* __restrict__ out, const std::uint32_t* __restrict__ idx,
                                     const T* __restrict__ vals, std::uint32_t n_scatter) {
    // grid=1, block=1 — deterministic index-order last-wins.
    for (std::uint32_t j = 0; j < n_scatter; ++j) out[idx[j]] = vals[j];
}

// scatter_add: out starts as a copy of base; out[idx[j]] += vals[j] in index
// order (accumulate). Serial single-pass (deterministic; scatter counts small).
template <class T>
__global__ void k_scatter_add_serial(T* __restrict__ out, const std::uint32_t* __restrict__ idx,
                                     const T* __restrict__ vals, std::uint32_t n_scatter) {
    for (std::uint32_t j = 0; j < n_scatter; ++j) out[idx[j]] = (T)(out[idx[j]] + vals[j]);
}

// ───────────────────────── reduce / scan (row-local) ─────────────────────

enum class RedKind : std::uint8_t { Sum, Max, Min };

// One CTA per row; grid-stride load over the row, then a shared-memory tree
// reduce. `len` may exceed the block width.
template <class T>
__global__ void k_reduce(const T* __restrict__ in, T* __restrict__ out,
                         std::uint32_t rows, std::uint32_t len, RedKind kind) {
    __shared__ T sh[kTier0Block];
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const T* r = in + (std::uint64_t)row * len;

    T seed0 = r[threadIdx.x < len ? threadIdx.x : 0];
    T acc = (kind == RedKind::Sum) ? (T)0 : seed0;
    for (std::uint32_t i = threadIdx.x; i < len; i += blockDim.x) {
        T v = r[i];
        if (kind == RedKind::Sum) acc = (T)(acc + v);
        else if (kind == RedKind::Max) acc = (v > acc ? v : acc);
        else acc = (v < acc ? v : acc);
    }
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (std::uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            T a = sh[threadIdx.x], b = sh[threadIdx.x + s];
            if (kind == RedKind::Sum) sh[threadIdx.x] = (T)(a + b);
            else if (kind == RedKind::Max) sh[threadIdx.x] = (b > a ? b : a);
            else sh[threadIdx.x] = (b < a ? b : a);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = sh[0];
}

// reduce_argmax row-local (float rows). Ties resolve to the LOWER index (pinned
// with echo in P0 open-Q #4). Output is a u32 index per row.
__global__ void k_reduce_argmax(const float* __restrict__ in, std::uint32_t* __restrict__ out,
                                std::uint32_t rows, std::uint32_t len) {
    __shared__ float sh_v[kTier0Block];
    __shared__ std::uint32_t sh_i[kTier0Block];
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float* r = in + (std::uint64_t)row * len;

    float best = t0_neg_inf();
    std::uint32_t bi = 0;
    for (std::uint32_t i = threadIdx.x; i < len; i += blockDim.x) {
        float v = r[i];
        if (v > best) { best = v; bi = i; }
    }
    sh_v[threadIdx.x] = best;
    sh_i[threadIdx.x] = bi;
    __syncthreads();
    for (std::uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float ov = sh_v[threadIdx.x + s];
            std::uint32_t oi = sh_i[threadIdx.x + s];
            if (ov > sh_v[threadIdx.x] || (ov == sh_v[threadIdx.x] && oi < sh_i[threadIdx.x])) {
                sh_v[threadIdx.x] = ov;
                sh_i[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = sh_i[0];
}

enum class ScanKind : std::uint8_t { Sum, Prod };

// Inclusive scan per row (cumsum/cumprod). One CTA per row; tiled Hillis-Steele
// with a running block carry so `len` may exceed the block width.
template <class T>
__global__ void k_scan(const T* __restrict__ in, T* __restrict__ out,
                       std::uint32_t rows, std::uint32_t len, ScanKind kind) {
    __shared__ T sh[kTier0Block];
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const T* ri = in + (std::uint64_t)row * len;
    T* ro = out + (std::uint64_t)row * len;

    T carry = (kind == ScanKind::Sum) ? (T)0 : (T)1;
    for (std::uint32_t base = 0; base < len; base += blockDim.x) {
        std::uint32_t i = base + threadIdx.x;
        T v = (i < len) ? ri[i] : ((kind == ScanKind::Sum) ? (T)0 : (T)1);
        sh[threadIdx.x] = v;
        __syncthreads();
        for (std::uint32_t off = 1; off < blockDim.x; off <<= 1) {
            T add = (threadIdx.x >= off) ? sh[threadIdx.x - off]
                                         : ((kind == ScanKind::Sum) ? (T)0 : (T)1);
            __syncthreads();
            sh[threadIdx.x] = (kind == ScanKind::Sum) ? (T)(sh[threadIdx.x] + add)
                                                      : (T)(sh[threadIdx.x] * add);
            __syncthreads();
        }
        if (i < len) {
            ro[i] = (kind == ScanKind::Sum) ? (T)(sh[threadIdx.x] + carry)
                                            : (T)(sh[threadIdx.x] * carry);
        }
        // carry := combine(carry, tile total) = value at last lane of this tile.
        T tile_total = sh[blockDim.x - 1];
        __syncthreads();
        carry = (kind == ScanKind::Sum) ? (T)(carry + tile_total) : (T)(carry * tile_total);
    }
}

// ────────────────────────── normalize (row-local) ────────────────────────

enum class NormKind : std::uint8_t { Softmax, LogSoftmax, L2Norm };

// One CTA per row: reduce (max/sumsq) → map. Numerically stable softmax.
__global__ void k_normalize(const float* __restrict__ in, float* __restrict__ out,
                            std::uint32_t rows, std::uint32_t len, NormKind kind) {
    __shared__ float sh[kTier0Block];
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float* ri = in + (std::uint64_t)row * len;
    float* ro = out + (std::uint64_t)row * len;

    auto block_reduce = [&](float v, bool is_max) -> float {
        sh[threadIdx.x] = v;
        __syncthreads();
        for (std::uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                float o = sh[threadIdx.x + s];
                sh[threadIdx.x] = is_max ? fmaxf(sh[threadIdx.x], o) : (sh[threadIdx.x] + o);
            }
            __syncthreads();
        }
        float r = sh[0];
        __syncthreads();
        return r;
    };

    if (kind == NormKind::L2Norm) {
        float local = 0.f;
        for (std::uint32_t i = threadIdx.x; i < len; i += blockDim.x) { float v = ri[i]; local += v * v; }
        float ss = block_reduce(local, /*is_max=*/false);
        float inv = rsqrtf(ss);
        for (std::uint32_t i = threadIdx.x; i < len; i += blockDim.x) ro[i] = ri[i] * inv;
        return;
    }

    // softmax / log_softmax
    float local_max = t0_neg_inf();
    for (std::uint32_t i = threadIdx.x; i < len; i += blockDim.x) local_max = fmaxf(local_max, ri[i]);
    float m = block_reduce(local_max, /*is_max=*/true);

    float local_sum = 0.f;
    for (std::uint32_t i = threadIdx.x; i < len; i += blockDim.x) local_sum += expf(ri[i] - m);
    float sum = block_reduce(local_sum, /*is_max=*/false);

    if (kind == NormKind::Softmax) {
        float inv = 1.f / sum;
        for (std::uint32_t i = threadIdx.x; i < len; i += blockDim.x) ro[i] = expf(ri[i] - m) * inv;
    } else {  // LogSoftmax
        float lse = m + logf(sum);
        for (std::uint32_t i = threadIdx.x; i < len; i += blockDim.x) ro[i] = ri[i] - lse;
    }
}

// ──────────────────────────────── sampling ───────────────────────────────

// mask_apply(logits, mask): out[j] = mask[j] ? logits[j] : -inf. Unpacked bool
// mask (materialized tier-0 form; the packed wire bitset is a D1 transport
// detail the runtime unpacks). Grid-stride over numel.
__global__ void k_mask_apply(const float* __restrict__ logits, const std::uint8_t* __restrict__ mask,
                             float* __restrict__ out, std::uint64_t n) {
    for (std::uint64_t i = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x;
         i < n; i += (std::uint64_t)gridDim.x * blockDim.x) {
        out[i] = mask[i] ? logits[i] : t0_neg_inf();
    }
}

// gumbel(rng, [rows, len]): fresh Gumbel noise g[r,j] from the ambient per-row
// seed S[r] and stream salt. Model B — no seed operand; the executor binds
// row_seeds. Grid-stride over rows*len.
__global__ void k_gumbel(const std::uint32_t* __restrict__ row_seed, std::uint32_t stream,
                         float* __restrict__ out, std::uint32_t rows, std::uint32_t len) {
    std::uint64_t n = (std::uint64_t)rows * len;
    for (std::uint64_t t = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x;
         t < n; t += (std::uint64_t)gridDim.x * blockDim.x) {
        std::uint32_t row = (std::uint32_t)(t / len);
        int col = (int)(t % len);
        unsigned long long se = t0_seed_eff_stream(row_seed[row], stream);
        out[t] = t0_gumbel_noise(se, col);
    }
}

// rng (0x70, ambient seed): per-row draw from S[r] + stream salt. `gumbel`=1 →
// Gumbel noise, else uniform in (0,1). Bit-parity with sample_temp.cu.
__global__ void k_rng_ambient(const std::uint32_t* __restrict__ row_seed, std::uint32_t stream,
                              float* __restrict__ out, std::uint32_t rows, std::uint32_t len, int gumbel) {
    std::uint64_t n = (std::uint64_t)rows * len;
    for (std::uint64_t t = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x;
         t < n; t += (std::uint64_t)gridDim.x * blockDim.x) {
        std::uint32_t row = (std::uint32_t)(t / len);
        int col = (int)(t % len);
        unsigned long long se = t0_seed_eff_stream(row_seed[row], stream);
        float u = t0_hash_uniform(se, col);
        out[t] = gumbel ? -logf(-logf(u)) : u;
    }
}

// rng_keyed (0x71, state = [key, ctr] U32): seed64 = splitmix64((key<<32)|ctr);
// element j (flat row-major over the shape) draws u = hash_uniform(seed64, j).
// `gumbel`=1 → -log(-log(u)). Pure function of (key, ctr, j) — replay-exact.
__global__ void k_rng_keyed(const std::uint32_t* __restrict__ state, float* __restrict__ out,
                            std::uint64_t numel, int gumbel) {
    unsigned long long seed64 = t0_splitmix64(((unsigned long long)state[0] << 32) | (unsigned long long)state[1]);
    for (std::uint64_t j = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x;
         j < numel; j += (std::uint64_t)gridDim.x * blockDim.x) {
        float u = t0_hash_uniform(seed64, (int)j);
        out[j] = gumbel ? -logf(-logf(u)) : u;
    }
}

// mask_apply_packed(logits, packed_mask): out[.., c] = bit_c(mask) ? logits[.., c] : -inf.
// Per echo's pinned matrix semantics (8cca6430): ONE packed word-row
// `mask [ceil(len/32)]`, BROADCAST across rows — the bit index is the last-axis
// COLUMN `c = flat_index % len`, NEVER the flat element or a per-row mask. Per-row
// DISTINCT masks are the composed bool form (select), not this op. 1 = ALLOWED.
// `mask_words` = ceil(len/32) (the single mask row's size). Grid-stride over rows*len.
__global__ void k_mask_apply_packed(const float* __restrict__ logits, const std::uint32_t* __restrict__ mask,
                                    float* __restrict__ out, std::uint32_t rows, std::uint32_t len,
                                    std::uint32_t mask_words) {
    std::uint64_t n = (std::uint64_t)rows * len;
    for (std::uint64_t t = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x;
         t < n; t += (std::uint64_t)gridDim.x * blockDim.x) {
        std::uint32_t r = (std::uint32_t)(t / len);   // row
        std::uint32_t j = (std::uint32_t)(t % len);   // last-axis column c
        std::uint32_t word = mask[(std::uint64_t)r * mask_words + (j >> 5)];  // per-row mask block
        bool keep = (word >> (j & 31)) & 1u;
        out[t] = keep ? logits[t] : t0_neg_inf();
    }
}

// ────────────────────────────── order family ─────────────────────────────

// rank_le(x, k): per-row bool, true iff x's rank (0 = largest) < k, i.e. fewer
// than k elements are strictly greater. Ties may select > k entries — the exact
// tie/NaN contract is echo-pinned (P0 open-Q #4). One CTA per row.
__global__ void k_rank_le(const float* __restrict__ in, std::uint8_t* __restrict__ out,
                          std::uint32_t rows, std::uint32_t len, std::uint32_t k) {
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float* r = in + (std::uint64_t)row * len;
    std::uint8_t* o = out + (std::uint64_t)row * len;
    for (std::uint32_t i = threadIdx.x; i < len; i += blockDim.x) {
        float v = r[i];
        std::uint32_t greater = 0;
        for (std::uint32_t j = 0; j < len; ++j) greater += (r[j] > v) ? 1u : 0u;
        o[i] = (greater < k) ? 1u : 0u;
    }
}

// pivot_threshold(score, rank_le(k)): keep entries whose rank < k, others → -inf
// (the selection form used as an attn_page_mask / top-k truncation, §6.1). Same
// rank rule as k_rank_le. One CTA per row.
__global__ void k_pivot_threshold_rankle(const float* __restrict__ in, float* __restrict__ out,
                                         std::uint32_t rows, std::uint32_t len, std::uint32_t k) {
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float* r = in + (std::uint64_t)row * len;
    float* o = out + (std::uint64_t)row * len;
    for (std::uint32_t i = threadIdx.x; i < len; i += blockDim.x) {
        float v = r[i];
        std::uint32_t greater = 0;
        for (std::uint32_t j = 0; j < len; ++j) greater += (r[j] > v) ? 1u : 0u;
        o[i] = (greater < k) ? v : t0_neg_inf();
    }
}

// ─────────────────────── library kernels (top_k, matmul) ─────────────────
// Linked, not generated (T9/§7.3). Tier-0 provides a correct baseline; the perf
// path swaps in bitonic top_k / cuBLASLt matmul.

// top_k row-local: k iterations of block-argmax with a moving threshold. Emits
// values (desc) and their column indices. Ties break by ascending index (the
// T8 numeric contract). NO dynamic shared memory — an earlier version kept a
// `len`-byte "already-picked" mask in shared memory, which overflows the
// per-block shared limit for large rows (e.g. a flattened [B*V] beam top_k over
// the full vocab) and silently fails the launch. Instead we carry the last
// picked (value, index) as a threshold: an element i is still available iff it
// sorts strictly after the previous pick — `v < prev_v || (v == prev_v && i >
// prev_i)`. O(k·len) with only the fixed-size reduction scratch.
__global__ void k_topk_rows(const float* __restrict__ in, float* __restrict__ out_val,
                            std::uint32_t* __restrict__ out_idx,
                            std::uint32_t rows, std::uint32_t len, std::uint32_t k) {
    __shared__ float sh_v[kTier0Block];
    __shared__ std::uint32_t sh_i[kTier0Block];
    __shared__ float prev_v;
    __shared__ std::uint32_t prev_i;
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float* r = in + (std::uint64_t)row * len;

    // +inf threshold ⇒ the first pick sees every finite element as available.
    if (threadIdx.x == 0) { prev_v = t0_pos_inf(); prev_i = 0; }
    __syncthreads();

    for (std::uint32_t pick = 0; pick < k; ++pick) {
        const float pv = prev_v; const std::uint32_t pi = prev_i;
        float best = t0_neg_inf();
        std::uint32_t bi = 0xFFFFFFFFu;
        for (std::uint32_t i = threadIdx.x; i < len; i += blockDim.x) {
            const float v = r[i];
            // Available iff it sorts strictly after the previous pick.
            const bool avail = (v < pv) || (v == pv && i > pi);
            if (!avail) continue;
            if (v > best || (v == best && i < bi)) { best = v; bi = i; }
        }
        sh_v[threadIdx.x] = best; sh_i[threadIdx.x] = bi;
        __syncthreads();
        for (std::uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                float ov = sh_v[threadIdx.x + s]; std::uint32_t oi = sh_i[threadIdx.x + s];
                if (ov > sh_v[threadIdx.x] || (ov == sh_v[threadIdx.x] && oi < sh_i[threadIdx.x])) {
                    sh_v[threadIdx.x] = ov; sh_i[threadIdx.x] = oi;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            out_val[(std::uint64_t)row * k + pick] = sh_v[0];
            out_idx[(std::uint64_t)row * k + pick] = sh_i[0];
            prev_v = sh_v[0]; prev_i = sh_i[0];
        }
        __syncthreads();
    }
}

// matmul: naive [M,K] x [K,N] → [M,N], row-major. Library kernel.
__global__ void k_matmul(const float* __restrict__ A, const float* __restrict__ B,
                         float* __restrict__ C, std::uint32_t M, std::uint32_t K, std::uint32_t N) {
    std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t rowm = blockIdx.y;
    if (rowm >= M || col >= N) return;
    float acc = 0.f;
    for (std::uint32_t kk = 0; kk < K; ++kk) acc += A[(std::uint64_t)rowm * K + kk] * B[(std::uint64_t)kk * N + col];
    C[(std::uint64_t)rowm * N + col] = acc;
}

// broadcast — materialize a scalar or per-row value into [rows, len].
//   mode 0: scalar src[0]              → out[i] = src[0]
//   mode 1: per-row src[row]           → out[row*len + j] = src[row]  (RowBroadcast)
template <class T>
__global__ void k_broadcast(const T* __restrict__ src, T* __restrict__ out,
                            std::uint32_t rows, std::uint32_t len, int mode) {
    std::uint64_t n = (std::uint64_t)rows * len;
    for (std::uint64_t t = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x;
         t < n; t += (std::uint64_t)gridDim.x * blockDim.x) {
        out[t] = (mode == 0) ? src[0] : src[t / len];
    }
}

// General same-rank broadcast (shape family, §appendix): each source dim is 1 or
// equal to the target dim. `meta` = [target_dims(4), src_strides(4)] where a
// broadcasted dim (src dim 1, target > 1) has stride 0. out[i] decomposes into
// target coords (row-major) → source offset = Σ coord[d]·src_stride[d]. Handles
// scalar/per-row AND tiling (e.g. [1,1,P]→[B,P,PAGE]).
template <class T>
__global__ void k_broadcast_general(const T* __restrict__ src, T* __restrict__ out,
                                    const std::uint32_t* __restrict__ meta, std::uint32_t rank,
                                    std::uint64_t numel) {
    for (std::uint64_t i = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x;
         i < numel; i += (std::uint64_t)gridDim.x * blockDim.x) {
        std::uint64_t rem = i, soff = 0;
        // target dims are meta[0..rank) with dim 0 outermost; walk innermost→out.
        for (int d = (int)rank - 1; d >= 0; --d) {
            std::uint32_t td = meta[d];
            std::uint32_t coord = (std::uint32_t)(rem % td);
            rem /= td;
            soff += (std::uint64_t)coord * meta[4 + d];
        }
        out[i] = src[soff];
    }
}

// transpose — materialize [rows, cols] → [cols, rows].
template <class T>
__global__ void k_transpose(const T* __restrict__ src, T* __restrict__ out,
                            std::uint32_t rows, std::uint32_t cols) {
    std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;   // col
    std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;   // row
    if (x < cols && y < rows) out[(std::uint64_t)x * rows + y] = src[(std::uint64_t)y * cols + x];
}

}  // namespace pie_cuda_driver::ptir
