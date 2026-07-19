#pragma once

// PTIR tier-0 op-kernel library (overview §7.3 "Tier 0 — interpret").
// One prebuilt row-parallel CUDA kernel per core op
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
// Compiled by nvcc (part of the driver lib / the tier-0 test). Tier 1
// re-emits the same math fused per stage via NVRTC (sampling_ir codegen
// generalized); this library is the correctness baseline every tier diffs
// against the host golden interpreter.
//
// RNG PARITY: the gumbel/uniform helpers consume the generated PTIR RNG
// contract, preserving the shipped sample_temp bit mapping.

#include <cstdint>
#include <type_traits>

#include <cuda_runtime.h>
#include <math_constants.h>

#include "pie_native/ptir/op_table.hpp"
#include "pie_native/ptir/rng_contract.generated.h"

namespace pie_cuda_driver::pipeline {

inline constexpr int kTier0Block = 256;   // fixed CTA width for row-local kernels
inline constexpr int kCanonicalReduceWidth = 32;
inline constexpr int kCanonicalReduceLevels = 8;

// ───────────────────────────── device helpers ────────────────────────────

__device__ __forceinline__ float t0_pos_inf() { return __int_as_float(0x7f800000); }
__device__ __forceinline__ float t0_neg_inf() { return __int_as_float(0xff800000); }

// RNG — generated contract plus backend-specific transcendental spelling.
__device__ __forceinline__ unsigned long long t0_seed_eff(unsigned int seed_u32) {
    return ptir_rng_seed_eff(seed_u32);
}
__device__ __forceinline__ unsigned long long t0_stream_salt(unsigned int stream) {
    return ptir_rng_stream_salt(stream);
}
__device__ __forceinline__ unsigned long long t0_seed_eff_stream(unsigned int seed_u32,
                                                                 unsigned int stream) {
    return ptir_rng_seed_eff_stream(seed_u32, stream);
}
__device__ __forceinline__ float t0_hash_uniform(unsigned long long seed_eff, int j) {
    return ptir_rng_hash_uniform(seed_eff, static_cast<unsigned int>(j));
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
__device__ __forceinline__ T t0_unary_value(T value, UnKind kind) {
    if constexpr (std::is_same_v<T, float>) {
        switch (kind) {
            case UnKind::Neg: return -value;
            case UnKind::Exp: return expf(value);
            case UnKind::Log: return logf(value);
            case UnKind::Recip: return 1.0f / value;
            case UnKind::Abs: return fabsf(value);
            case UnKind::Sign:
                return static_cast<float>((value > 0.0f) - (value < 0.0f));
        }
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        const auto bits = static_cast<std::uint32_t>(value);
        const auto negated = static_cast<std::int32_t>(0u - bits);
        switch (kind) {
            case UnKind::Neg: return negated;
            case UnKind::Abs: return value < 0 ? negated : value;
            case UnKind::Sign:
                return static_cast<std::int32_t>(
                    (value > 0) - (value < 0));
            default: return value;
        }
    } else {
        switch (kind) {
            case UnKind::Neg: return static_cast<T>(0u - value);
            case UnKind::Abs: return value;
            case UnKind::Sign: return static_cast<T>(value != 0);
            default: return value;
        }
    }
    return value;
}

template <class T>
__global__ void k_unary(const T* __restrict__ a, T* __restrict__ out,
                        std::uint64_t n, UnKind kind) {
    for (std::uint64_t i = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x;
         i < n; i += (std::uint64_t)gridDim.x * blockDim.x) {
        out[i] = t0_unary_value(a[i], kind);
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

template <class TIn>
__global__ void k_cast_bool(
    const TIn* __restrict__ in,
    std::uint8_t* __restrict__ out,
    std::uint64_t n) {
    for (std::uint64_t i =
             blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         i < n;
         i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        out[i] = in[i] != static_cast<TIn>(0);
    }
}

// ───────────────────────────────── index ─────────────────────────────────

__global__ void k_iota(std::uint32_t* __restrict__ out, std::uint32_t n) {
    for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        out[i] = i;
    }
}

template <class Index>
__device__ __forceinline__ bool valid_axis0_index(
    Index value, std::uint32_t extent, std::uint32_t* index) {
    if constexpr (std::is_signed_v<Index>) {
        if (value < 0) return false;
    }
    const std::uint64_t widened = static_cast<std::uint64_t>(value);
    if (widened >= extent) return false;
    *index = static_cast<std::uint32_t>(widened);
    return true;
}

template <class T, class Index>
__global__ void k_gather_axis0(
    const T* __restrict__ src,
    const Index* __restrict__ indices,
    T* __restrict__ out,
    std::uint32_t index_count,
    std::uint32_t axis0,
    std::uint32_t inner) {
    const std::uint64_t numel =
        static_cast<std::uint64_t>(index_count) * inner;
    for (std::uint64_t flat =
             blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         flat < numel;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t position =
            static_cast<std::uint32_t>(flat / inner);
        const std::uint32_t offset =
            static_cast<std::uint32_t>(flat % inner);
        std::uint32_t selected = 0;
        out[flat] = valid_axis0_index(
                        indices[position], axis0, &selected)
            ? src[static_cast<std::uint64_t>(selected) * inner + offset]
            : T{};
    }
}

template <class T, class Index>
__global__ void k_gather_row(
    const T* __restrict__ src,
    const Index* __restrict__ indices,
    T* __restrict__ out,
    std::uint32_t rows,
    std::uint32_t columns) {
    for (std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < rows;
         row += gridDim.x * blockDim.x) {
        std::uint32_t column = 0;
        out[row] = valid_axis0_index(
                       indices[row], columns, &column)
            ? src[static_cast<std::uint64_t>(row) * columns + column]
            : T{};
    }
}

template <class T, class Index, bool Add>
__global__ void k_scatter_axis0_serial(
    T* __restrict__ out,
    const Index* __restrict__ indices,
    const T* __restrict__ vals,
    std::uint32_t index_count,
    std::uint32_t axis0,
    std::uint32_t inner,
    bool scalar_vals) {
    for (std::uint32_t update = 0; update < index_count; ++update) {
        std::uint32_t selected = 0;
        if (!valid_axis0_index(indices[update], axis0, &selected)) continue;
        for (std::uint32_t offset = 0; offset < inner; ++offset) {
            const T value = vals[
                scalar_vals
                    ? 0
                    : static_cast<std::uint64_t>(update) * inner + offset];
            T& destination =
                out[static_cast<std::uint64_t>(selected) * inner + offset];
            if constexpr (Add) {
                if constexpr (std::is_same_v<T, std::int32_t>) {
                    destination = static_cast<std::int32_t>(
                        static_cast<std::uint32_t>(destination) +
                        static_cast<std::uint32_t>(value));
                } else {
                    destination = static_cast<T>(destination + value);
                }
            } else {
                destination = value;
            }
        }
    }
}

// ───────────────────────── reduce / scan (row-local) ─────────────────────

enum class RedKind : std::uint8_t { Sum, Max, Min };

template <class T> struct RedLimits;
template <> struct RedLimits<float> {
    __device__ static float lowest() { return t0_neg_inf(); }
    __device__ static float highest() { return t0_pos_inf(); }
};
template <> struct RedLimits<std::int32_t> {
    __device__ static std::int32_t lowest() { return -2147483647 - 1; }
    __device__ static std::int32_t highest() { return 2147483647; }
};
template <> struct RedLimits<std::uint32_t> {
    __device__ static std::uint32_t lowest() { return 0; }
    __device__ static std::uint32_t highest() { return 0xffffffffu; }
};

template <class T>
__device__ __forceinline__ T red_identity(RedKind kind) {
    return kind == RedKind::Sum ? (T)0
         : kind == RedKind::Max ? RedLimits<T>::lowest()
                               : RedLimits<T>::highest();
}

template <class T>
__device__ __forceinline__ T red_combine(T left, T right, RedKind kind) {
    if (kind == RedKind::Sum) {
        if constexpr (std::is_same_v<T, std::int32_t>) {
            return static_cast<std::int32_t>(
                static_cast<std::uint32_t>(left) +
                static_cast<std::uint32_t>(right));
        }
        return static_cast<T>(left + right);
    }
    if (kind == RedKind::Max) return right > left ? right : left;
    return right < left ? right : left;
}

template <>
__device__ __forceinline__ float red_combine<float>(
    float left, float right, RedKind kind) {
    if (kind == RedKind::Sum) return left + right;
    const bool left_nan = isnan(left);
    const bool right_nan = isnan(right);
    if (left_nan) return right_nan ? red_identity<float>(kind) : right;
    if (right_nan) return left;
    if (left == 0.0f && right == 0.0f) {
        const std::uint32_t left_sign = __float_as_uint(left) & 0x80000000u;
        const std::uint32_t right_sign = __float_as_uint(right) & 0x80000000u;
        return __uint_as_float(
            kind == RedKind::Max ? left_sign & right_sign
                                 : left_sign | right_sign);
    }
    if (kind == RedKind::Max) return fmaxf(left, right);
    return fminf(left, right);
}

template <class T>
__device__ __forceinline__ T reduce_canonical_slot(
    T* slot, std::uint32_t lane, std::uint32_t count, RedKind kind) {
    if (lane >= count) slot[lane] = red_identity<T>(kind);
    __syncwarp();
    for (std::uint32_t offset = 16; offset > 0; offset >>= 1) {
        if (lane < offset) {
            slot[lane] = red_combine(slot[lane], slot[lane + offset], kind);
        }
        __syncwarp();
    }
    return slot[0];
}

// One 32-lane logical tree per row. Tile partials are streamed through a
// base-32 carry stack, reproducing the recursive 32-wide schedule without
// launch-configuration-dependent accumulation or vocabulary-sized scratch.
template <class T>
__global__ void k_reduce(const T* __restrict__ in, T* __restrict__ out,
                         std::uint32_t rows, std::uint32_t len, RedKind kind) {
    __shared__ T levels[kCanonicalReduceLevels][kCanonicalReduceWidth];
    __shared__ T tile[kCanonicalReduceWidth];
    __shared__ std::uint32_t counts[kCanonicalReduceLevels];
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const std::uint32_t lane = threadIdx.x;
    const T* r = in + (std::uint64_t)row * len;

    if (lane < kCanonicalReduceLevels) counts[lane] = 0;
    __syncwarp();
    for (std::uint32_t base = 0; base < len; base += kCanonicalReduceWidth) {
        const std::uint32_t index = base + lane;
        tile[lane] = index < len ? r[index] : red_identity<T>(kind);
        const std::uint32_t tile_count =
            len - base < kCanonicalReduceWidth ? len - base : kCanonicalReduceWidth;
        const T partial = reduce_canonical_slot(tile, lane, tile_count, kind);
        if (lane == 0) {
            levels[0][counts[0]++] = partial;
        }
        __syncwarp();
        for (std::uint32_t level = 0; level + 1 < kCanonicalReduceLevels; ++level) {
            if (counts[level] != kCanonicalReduceWidth) break;
            const T carry = reduce_canonical_slot(
                levels[level], lane, kCanonicalReduceWidth, kind);
            if (lane == 0) {
                counts[level] = 0;
                levels[level + 1][counts[level + 1]++] = carry;
            }
            __syncwarp();
        }
    }

    for (std::uint32_t level = 0; level + 1 < kCanonicalReduceLevels; ++level) {
        const std::uint32_t count = counts[level];
        if (count == 0) continue;
        bool higher = false;
        for (std::uint32_t next = level + 1; next < kCanonicalReduceLevels; ++next) {
            higher = higher || counts[next] != 0;
        }
        if (count == 1 && !higher) {
            if (lane == 0) out[row] = levels[level][0];
            return;
        }
        const T carry = reduce_canonical_slot(levels[level], lane, count, kind);
        if (lane == 0) {
            counts[level] = 0;
            levels[level + 1][counts[level + 1]++] = carry;
        }
        __syncwarp();
    }
    if (lane == 0) {
        out[row] = counts[kCanonicalReduceLevels - 1] == 0
            ? red_identity<T>(kind)
            : levels[kCanonicalReduceLevels - 1][0];
    }
}

// reduce_argmax row-local. Integer comparisons stay in their declared dtype;
// float NaNs are never selected. Ties resolve to the LOWER index.
template <class T>
__global__ void k_reduce_argmax(const T* __restrict__ in,
                                std::uint32_t* __restrict__ out,
                                std::uint32_t rows, std::uint32_t len) {
    __shared__ T sh_v[kTier0Block];
    __shared__ std::uint32_t sh_i[kTier0Block];
    __shared__ std::uint8_t sh_have[kTier0Block];
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const T* r = in + (std::uint64_t)row * len;

    T best{};
    std::uint32_t bi = 0;
    bool have = false;
    for (std::uint32_t i = threadIdx.x; i < len; i += blockDim.x) {
        const T v = r[i];
        bool selectable = true;
        if constexpr (std::is_same_v<T, float>) {
            selectable = !isnan(v);
        }
        if (selectable &&
            (!have || v > best || (v == best && i < bi))) {
            best = v;
            bi = i;
            have = true;
        }
    }
    sh_v[threadIdx.x] = best;
    sh_i[threadIdx.x] = bi;
    sh_have[threadIdx.x] = have;
    __syncthreads();
    for (std::uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const T ov = sh_v[threadIdx.x + s];
            std::uint32_t oi = sh_i[threadIdx.x + s];
            const bool other_have = sh_have[threadIdx.x + s] != 0;
            const bool self_have = sh_have[threadIdx.x] != 0;
            if (other_have &&
                (!self_have || ov > sh_v[threadIdx.x] ||
                 (ov == sh_v[threadIdx.x] && oi < sh_i[threadIdx.x]))) {
                sh_v[threadIdx.x] = ov;
                sh_i[threadIdx.x] = oi;
                sh_have[threadIdx.x] = 1;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = sh_have[0] ? sh_i[0] : 0;
}

enum class ScanKind : std::uint8_t { Sum, Prod };

template <class T>
__device__ __forceinline__ T scan_combine(
    T left, T right, ScanKind kind) {
    if constexpr (std::is_same_v<T, std::int32_t>) {
        const auto a = static_cast<std::uint32_t>(left);
        const auto b = static_cast<std::uint32_t>(right);
        return static_cast<std::int32_t>(
            kind == ScanKind::Sum ? a + b : a * b);
    }
    return kind == ScanKind::Sum
        ? static_cast<T>(left + right)
        : static_cast<T>(left * right);
}

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
            sh[threadIdx.x] =
                scan_combine(sh[threadIdx.x], add, kind);
            __syncthreads();
        }
        if (i < len) {
            ro[i] = scan_combine(carry, sh[threadIdx.x], kind);
        }
        // carry := combine(carry, tile total) = value at last lane of this tile.
        T tile_total = sh[blockDim.x - 1];
        __syncthreads();
        carry = scan_combine(carry, tile_total, kind);
    }
}

// ──────────────────────────────── sampling ───────────────────────────────

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
    unsigned long long seed64 = ptir_rng_keyed_seed(state[0], state[1]);
    for (std::uint64_t j = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x;
         j < numel; j += (std::uint64_t)gridDim.x * blockDim.x) {
        float u = t0_hash_uniform(seed64, (int)j);
        out[j] = gumbel ? -logf(-logf(u)) : u;
    }
}

// mask_apply_packed(logits, packed_mask): out[.., c] = bit_c(mask) ? logits[.., c] : -inf.
// Per the pinned matrix semantics (8cca6430): ONE packed word-row
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

enum class Tier0StructuredMaskKind : std::uint8_t {
    Causal,
    SlidingWindow,
    SinkWindow,
};

__device__ __forceinline__ bool structured_position_allows(
    std::uint32_t position,
    std::uint32_t key,
    Tier0StructuredMaskKind kind,
    std::uint32_t window,
    std::uint32_t sink) {
    if (key > position) return false;
    if (kind == Tier0StructuredMaskKind::Causal) return true;
    const std::uint32_t key_plus_window =
        window > UINT32_MAX - key ? UINT32_MAX : key + window;
    const bool in_window = key_plus_window > position;
    return kind == Tier0StructuredMaskKind::SlidingWindow
        ? in_window
        : key < sink || in_window;
}

__global__ void k_structured_position_mask(
    const std::uint32_t* __restrict__ positions,
    std::uint8_t* __restrict__ out,
    std::uint32_t position_count,
    std::uint32_t key_count,
    Tier0StructuredMaskKind kind,
    std::uint32_t window,
    std::uint32_t sink) {
    const std::uint64_t numel =
        static_cast<std::uint64_t>(position_count) * key_count;
    for (std::uint64_t flat =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         flat < numel;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t position_index =
            static_cast<std::uint32_t>(flat / key_count);
        const std::uint32_t key =
            static_cast<std::uint32_t>(flat % key_count);
        out[flat] = structured_position_allows(
            positions[position_index], key, kind, window, sink);
    }
}

// ────────────────────────────── order family ─────────────────────────────

// ──────────────────── pivot_threshold predicates (dynamic) ───────────────
// The three `pivot_threshold` predicates (interface/ptir interp.rs eval_op,
// Op::PivotThreshold): the payload is ALWAYS a trace value (scalar or
// per-row [rows] vector), never a host immediate — resolved by the runner to
// a device buffer + dtype + numel (tier0_runner.hpp build_launch) and read
// here at launch time. `pred_numel<=1` broadcasts index 0 to every row
// (mirrors interp.rs `pick(len, r)`), else it's one value per row.

// Shared total order used by CummassLe's descending selection loop:
// descending value, ties → lower original index first, NaN sorts last (ties
// among NaNs → lower index first) — exactly `sort_desc_order`'s comparator.
// Returns true iff (av,ai) sorts STRICTLY BEFORE (bv,bi) in that order.
__device__ __forceinline__ bool t0_desc_before(float av, std::uint32_t ai, bool a_nan,
                                               float bv, std::uint32_t bi, bool b_nan) {
    if (!a_nan && !b_nan) return (av > bv) || (av == bv && ai < bi);
    if (a_nan && b_nan) return ai < bi;
    return !a_nan;   // non-NaN always sorts before NaN
}

// rank_le predicate inside pivot_threshold: `k` is a dynamic (scalar or
// per-row) I32/U32 trace value — NOT a host immediate (unlike the standalone
// `rank_le` op above). Ties/NaN contract mirrors interp.rs exactly: a NaN
// element is NEVER selected (interp.rs `if xi.is_nan() { continue; }` leaves
// its `keep` bit at the default false), and NaN elements never count toward
// another element's `greater` tally. One CTA per row.
template <class KT>
__global__ void k_pivot_rankle(const float* __restrict__ in, std::uint8_t* __restrict__ out,
                               std::uint32_t rows, std::uint32_t len,
                               const KT* __restrict__ k_buf, std::uint32_t k_numel) {
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float* r = in + (std::uint64_t)row * len;
    std::uint8_t* o = out + (std::uint64_t)row * len;
    const std::int64_t k_raw = (std::int64_t)k_buf[(k_numel <= 1) ? 0u : row];
    const std::int64_t kk = k_raw < 0 ? 0 : (k_raw > (std::int64_t)len ? (std::int64_t)len : k_raw);
    for (std::uint32_t i = threadIdx.x; i < len; i += blockDim.x) {
        const float v = r[i];
        if (isnan(v)) { o[i] = 0u; continue; }
        std::int64_t greater = 0;
        for (std::uint32_t j = 0; j < len; ++j) {
            const float y = r[j];
            if (!isnan(y) && y > v) ++greater;
        }
        o[i] = (greater < kk) ? 1u : 0u;
    }
}

// prob_ge predicate: elementwise `x[i] >= thr`, `thr` a dynamic (scalar or
// per-row) F32 trace value. Purely elementwise — grid-stride over rows*len.
__global__ void k_pivot_probge(const float* __restrict__ in, std::uint8_t* __restrict__ out,
                               std::uint32_t rows, std::uint32_t len,
                               const float* __restrict__ thr_buf, std::uint32_t thr_numel) {
    const std::uint64_t n = (std::uint64_t)rows * len;
    for (std::uint64_t t = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x;
         t < n; t += (std::uint64_t)gridDim.x * blockDim.x) {
        const std::uint32_t row = (std::uint32_t)(len == 0 ? 0 : t / len);
        const float thr = thr_buf[(thr_numel <= 1) ? 0u : row];
        out[t] = (in[t] >= thr) ? 1u : 0u;
    }
}

// cummass_le predicate (top-p / nucleus): keep the descending prefix whose
// EXCLUSIVE cumulative mass stays `< p` (interp.rs: `k[i] = excl < p; excl +=
// row[i]`), `p` a dynamic (scalar or per-row) F32 trace value. This is a CTA
// selection loop — one block-wide "next largest still-unpicked element" pick
// per iteration (the same incremental-threshold technique as `k_topk_rows`,
// generalized to the NaN-aware total order above) — instead of a `len`-way
// per-element rank pass (which would cost O(len^2) unconditionally, i.e. ~2.3e10
// ops at the 151936-token vocab this must stay practical for). Real LM
// distributions are peaked, so the loop typically stops after a handful of
// picks once the running mass clears `p`; pathological (near-uniform, p→1)
// rows still complete correctly, just in up to `len` picks (matching the
// dense case any correct implementation must cover). `out` is zero-inited so
// unvisited (excluded) elements default to false.
__global__ void k_pivot_cummassle(const float* __restrict__ in, std::uint8_t* __restrict__ out,
                                  std::uint32_t rows, std::uint32_t len,
                                  const float* __restrict__ p_buf, std::uint32_t p_numel) {
    __shared__ float sh_v[kTier0Block];
    __shared__ std::uint32_t sh_i[kTier0Block];
    __shared__ std::uint8_t sh_nan[kTier0Block];
    __shared__ float prev_v;
    __shared__ std::uint32_t prev_i;
    __shared__ std::uint8_t prev_nan;
    __shared__ float excl;
    __shared__ std::uint8_t stop;
    constexpr std::uint32_t kNone = 0xFFFFFFFFu;

    const std::uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float* r = in + (std::uint64_t)row * len;
    std::uint8_t* o = out + (std::uint64_t)row * len;
    const float p = p_buf[(p_numel <= 1) ? 0u : row];

    for (std::uint32_t i = threadIdx.x; i < len; i += blockDim.x) o[i] = 0u;
    if (threadIdx.x == 0) {
        prev_v = t0_pos_inf(); prev_i = 0u; prev_nan = 0u;   // sentinel: before everything
        excl = 0.0f; stop = 0u;
    }
    __syncthreads();

    for (std::uint32_t pick = 0; pick < len; ++pick) {
        if (stop) break;
        const float pv = prev_v; const std::uint32_t pi = prev_i; const bool p_nan = prev_nan != 0u;
        float best = 0.0f; std::uint32_t bi = kNone; bool best_nan = false;
        for (std::uint32_t i = threadIdx.x; i < len; i += blockDim.x) {
            const float v = r[i];
            const bool v_nan = isnan(v);
            // Available iff it sorts strictly after the previous pick — the
            // very first pick's `prev` (+inf, i=0, non-NaN) admits every i
            // (any real value or NaN sorts after +inf).
            if (!t0_desc_before(pv, pi, p_nan, v, i, v_nan)) continue;
            if (bi == kNone || t0_desc_before(v, i, v_nan, best, bi, best_nan)) {
                best = v; bi = i; best_nan = v_nan;
            }
        }
        sh_v[threadIdx.x] = best; sh_i[threadIdx.x] = bi; sh_nan[threadIdx.x] = best_nan ? 1u : 0u;
        __syncthreads();
        for (std::uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                const std::uint32_t oi = sh_i[threadIdx.x + s];
                if (oi != kNone &&
                    (sh_i[threadIdx.x] == kNone ||
                     t0_desc_before(sh_v[threadIdx.x + s], oi, sh_nan[threadIdx.x + s] != 0u,
                                    sh_v[threadIdx.x], sh_i[threadIdx.x], sh_nan[threadIdx.x] != 0u))) {
                    sh_v[threadIdx.x] = sh_v[threadIdx.x + s];
                    sh_i[threadIdx.x] = oi;
                    sh_nan[threadIdx.x] = sh_nan[threadIdx.x + s];
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const std::uint32_t idx = sh_i[0];
            if (idx == kNone) {
                stop = 1u;   // exhausted the row (len picks already done otherwise)
            } else if (excl < p) {
                o[idx] = 1u;
                excl += r[idx];
                prev_v = sh_v[0]; prev_i = idx; prev_nan = sh_nan[0];
            } else {
                stop = 1u;   // mass condition failed — descending order ⇒ all remaining fail too
            }
        }
        __syncthreads();
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
// picked total-order key as a threshold. Explicit validity admits +inf at index
// zero on the first iteration; NaNs sort last; ties (including signed zero)
// retain ascending source index. O(k·len) with fixed-size reduction scratch.
__global__ void k_topk_rows(const float* __restrict__ in, float* __restrict__ out_val,
                            std::uint32_t* __restrict__ out_idx,
                            std::uint32_t rows, std::uint32_t len, std::uint32_t k) {
    __shared__ float sh_v[kTier0Block];
    __shared__ std::uint32_t sh_i[kTier0Block];
    __shared__ std::uint8_t sh_nan[kTier0Block];
    __shared__ float prev_v;
    __shared__ std::uint32_t prev_i;
    __shared__ std::uint8_t prev_nan;
    __shared__ std::uint8_t prev_valid;
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float* r = in + (std::uint64_t)row * len;

    if (threadIdx.x == 0) {
        prev_v = 0.0f;
        prev_i = 0;
        prev_nan = 0;
        prev_valid = 0;
    }
    __syncthreads();

    for (std::uint32_t pick = 0; pick < k; ++pick) {
        const float pv = prev_v;
        const std::uint32_t pi = prev_i;
        const bool p_nan = prev_nan != 0;
        const bool p_valid = prev_valid != 0;
        float best = 0.0f;
        std::uint32_t bi = 0xFFFFFFFFu;
        bool best_nan = false;
        for (std::uint32_t i = threadIdx.x; i < len; i += blockDim.x) {
            const float v = r[i];
            const bool v_nan = isnan(v);
            const bool avail =
                !p_valid || t0_desc_before(
                    pv, pi, p_nan, v, i, v_nan);
            if (!avail) continue;
            if (bi == 0xFFFFFFFFu ||
                t0_desc_before(v, i, v_nan, best, bi, best_nan)) {
                best = v;
                bi = i;
                best_nan = v_nan;
            }
        }
        sh_v[threadIdx.x] = best;
        sh_i[threadIdx.x] = bi;
        sh_nan[threadIdx.x] = best_nan;
        __syncthreads();
        for (std::uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                const std::uint32_t oi = sh_i[threadIdx.x + s];
                if (oi != 0xFFFFFFFFu &&
                    (sh_i[threadIdx.x] == 0xFFFFFFFFu ||
                     t0_desc_before(
                         sh_v[threadIdx.x + s], oi,
                         sh_nan[threadIdx.x + s] != 0,
                         sh_v[threadIdx.x], sh_i[threadIdx.x],
                         sh_nan[threadIdx.x] != 0))) {
                    sh_v[threadIdx.x] = sh_v[threadIdx.x + s];
                    sh_i[threadIdx.x] = oi;
                    sh_nan[threadIdx.x] = sh_nan[threadIdx.x + s];
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            out_val[(std::uint64_t)row * k + pick] = sh_v[0];
            out_idx[(std::uint64_t)row * k + pick] = sh_i[0];
            prev_v = sh_v[0];
            prev_i = sh_i[0];
            prev_nan = sh_nan[0];
            prev_valid = 1;
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

}  // namespace pie_cuda_driver::pipeline
