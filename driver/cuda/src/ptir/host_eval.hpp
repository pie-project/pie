#pragma once

// PTIR tier-0 HOST reference evaluator — a pure-host mirror of the tier-0 CUDA
// kernels (tier0_kernels.cuh), computing each op's result on std::vectors with
// byte-identical semantics (same RNG constants, same argmax tie-break, same
// numerically-stable softmax reduction order where it matters).
//
// ROLE: this is charlie's SELF-CHECK oracle for the tier-0 kernels while echo's
// canonical host golden interpreter (the real conformance oracle, thrust-3 P4.1)
// is in flight. Once echo's interpreter + golden vectors land, every op is
// diffed against THOSE; this file remains a fast local cross-check. It is not
// the spec oracle — echo's is.
//
// Pure host C++: no CUDA, no driver deps. Included by the tier-0 test harness.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "ptir/op_table.hpp"
#include "ptir/tier0_kernels.cuh"  // for the BinKind/UnKind/... enums (host-safe)

namespace pie_cuda_driver::ptir::host_eval {

// ── RNG parity (mirrors t0_* in tier0_kernels.cuh, sample_temp.cu lineage) ──
inline std::uint64_t h_splitmix64(std::uint64_t x) {
    x ^= x >> 27; x *= 0x3C79AC492BA7B653ULL;
    x ^= x >> 33; x *= 0x1C69B3F74AC4AE35ULL;
    x ^= x >> 27;
    return x;
}
inline std::uint64_t h_seed_eff(std::uint32_t s) { return (std::uint64_t)s ^ 0xA5A5A5A5ULL; }
inline std::uint64_t h_stream_salt(std::uint32_t stream) {
    return h_splitmix64((std::uint64_t)stream * 0x9E3779B97F4A7C15ULL);
}
inline std::uint64_t h_seed_eff_stream(std::uint32_t s, std::uint32_t stream) {
    return h_seed_eff(s) ^ h_stream_salt(stream);
}
inline float h_hash_uniform(std::uint64_t seed_eff, int j) {
    std::uint64_t x = seed_eff + 0x9E3779B97F4A7C15ULL * (std::uint64_t)(j + 1);
    x = h_splitmix64(x);
    std::uint32_t bits = (std::uint32_t)(x >> 40);
    return ((float)bits + 0.5f) * (1.0f / 16777216.0f);
}
inline float h_gumbel_noise(std::uint64_t seed_eff, int j) {
    float u = h_hash_uniform(seed_eff, j);
    return -std::log(-std::log(u));
}

inline float neg_inf() { return -std::numeric_limits<float>::infinity(); }

// ─────────────────────────── map / element-wise ──────────────────────────
template <class T>
std::vector<T> binary(BinKind k, const std::vector<T>& a, const std::vector<T>& b) {
    std::vector<T> o(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        switch (k) {
            case BinKind::Add: o[i] = a[i] + b[i]; break;
            case BinKind::Sub: o[i] = a[i] - b[i]; break;
            case BinKind::Mul: o[i] = a[i] * b[i]; break;
            case BinKind::Div: o[i] = a[i] / b[i]; break;
            case BinKind::Rem:
                if constexpr (std::is_floating_point_v<T>) o[i] = std::fmod(a[i], b[i]);
                else o[i] = a[i] - (a[i] / b[i]) * b[i];
                break;
            case BinKind::MaxElem: o[i] = a[i] > b[i] ? a[i] : b[i]; break;
            case BinKind::MinElem: o[i] = a[i] < b[i] ? a[i] : b[i]; break;
        }
    }
    return o;
}
template <class T>
std::vector<T> unary(UnKind k, const std::vector<T>& a) {
    std::vector<T> o(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        switch (k) {
            case UnKind::Neg:   o[i] = -a[i]; break;
            case UnKind::Exp:   o[i] = (T)std::exp((float)a[i]); break;
            case UnKind::Log:   o[i] = (T)std::log((float)a[i]); break;
            case UnKind::Recip: o[i] = (T)(1.0f / (float)a[i]); break;
            case UnKind::Abs:   o[i] = (T)std::fabs((float)a[i]); break;
            case UnKind::Sign:  o[i] = (T)(((float)a[i] > 0.0f) - ((float)a[i] < 0.0f)); break;
        }
    }
    return o;
}
template <class T>
std::vector<std::uint8_t> compare(CmpKind k, const std::vector<T>& a, const std::vector<T>& b) {
    std::vector<std::uint8_t> o(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        bool r = false;
        switch (k) {
            case CmpKind::Eq: r = a[i] == b[i]; break;
            case CmpKind::Ne: r = a[i] != b[i]; break;
            case CmpKind::Lt: r = a[i] <  b[i]; break;
            case CmpKind::Le: r = a[i] <= b[i]; break;
            case CmpKind::Gt: r = a[i] >  b[i]; break;
            case CmpKind::Ge: r = a[i] >= b[i]; break;
        }
        o[i] = r ? 1u : 0u;
    }
    return o;
}
inline std::vector<std::uint8_t> logic(LogicKind k, const std::vector<std::uint8_t>& a,
                                       const std::vector<std::uint8_t>& b) {
    std::vector<std::uint8_t> o(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        std::uint8_t x = a[i] ? 1u : 0u, y = b[i] ? 1u : 0u;
        o[i] = (k == LogicKind::And) ? (x & y) : (x | y);
    }
    return o;
}
inline std::vector<std::uint8_t> logic_not(const std::vector<std::uint8_t>& a) {
    std::vector<std::uint8_t> o(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) o[i] = a[i] ? 0u : 1u;
    return o;
}
template <class T>
std::vector<T> select(const std::vector<std::uint8_t>& cond, const std::vector<T>& a,
                      const std::vector<T>& b) {
    std::vector<T> o(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) o[i] = cond[i] ? a[i] : b[i];
    return o;
}
template <class TIn, class TOut>
std::vector<TOut> cast(const std::vector<TIn>& in) {
    std::vector<TOut> o(in.size());
    for (std::size_t i = 0; i < in.size(); ++i) o[i] = (TOut)in[i];
    return o;
}

// ───────────────────────────────── index ─────────────────────────────────
inline std::vector<std::uint32_t> iota(std::uint32_t n) {
    std::vector<std::uint32_t> o(n);
    for (std::uint32_t i = 0; i < n; ++i) o[i] = i;
    return o;
}
template <class T>
std::vector<T> gather(const std::vector<T>& src, const std::vector<std::uint32_t>& idx) {
    std::vector<T> o(idx.size());
    for (std::size_t i = 0; i < idx.size(); ++i) o[i] = src[idx[i]];
    return o;
}
template <class T>
std::vector<T> gather_row(const std::vector<T>& src, const std::vector<std::uint32_t>& idx,
                          std::uint32_t row_len) {
    std::vector<T> o(idx.size() * row_len);
    for (std::size_t i = 0; i < idx.size(); ++i)
        for (std::uint32_t j = 0; j < row_len; ++j)
            o[i * row_len + j] = src[(std::size_t)idx[i] * row_len + j];
    return o;
}
template <class T>
std::vector<T> scatter_set(const std::vector<T>& base, const std::vector<std::uint32_t>& idx,
                           const std::vector<T>& vals) {
    std::vector<T> o = base;
    for (std::size_t j = 0; j < idx.size(); ++j) o[idx[j]] = vals[j];  // last wins
    return o;
}
template <class T>
std::vector<T> scatter_add(const std::vector<T>& base, const std::vector<std::uint32_t>& idx,
                           const std::vector<T>& vals) {
    std::vector<T> o = base;
    for (std::size_t j = 0; j < idx.size(); ++j) o[idx[j]] = (T)(o[idx[j]] + vals[j]);
    return o;
}

// ───────────────────────── reduce / scan (row-local) ─────────────────────
template <class T>
std::vector<T> reduce(RedKind k, const std::vector<T>& in, std::uint32_t rows, std::uint32_t len) {
    std::vector<T> o(rows);
    for (std::uint32_t r = 0; r < rows; ++r) {
        const T* row = in.data() + (std::size_t)r * len;
        T acc = (k == RedKind::Sum) ? (T)0 : row[0];
        for (std::uint32_t i = 0; i < len; ++i) {
            if (k == RedKind::Sum) acc = (T)(acc + row[i]);
            else if (k == RedKind::Max) acc = std::max(acc, row[i]);
            else acc = std::min(acc, row[i]);
        }
        o[r] = acc;
    }
    return o;
}
inline std::vector<std::uint32_t> reduce_argmax(const std::vector<float>& in, std::uint32_t rows,
                                                std::uint32_t len) {
    std::vector<std::uint32_t> o(rows);
    for (std::uint32_t r = 0; r < rows; ++r) {
        const float* row = in.data() + (std::size_t)r * len;
        float best = neg_inf();
        std::uint32_t bi = 0;
        for (std::uint32_t i = 0; i < len; ++i) if (row[i] > best) { best = row[i]; bi = i; }
        o[r] = bi;  // lower index on ties
    }
    return o;
}
template <class T>
std::vector<T> scan(ScanKind k, const std::vector<T>& in, std::uint32_t rows, std::uint32_t len) {
    std::vector<T> o(in.size());
    for (std::uint32_t r = 0; r < rows; ++r) {
        const T* ri = in.data() + (std::size_t)r * len;
        T* ro = o.data() + (std::size_t)r * len;
        T acc = (k == ScanKind::Sum) ? (T)0 : (T)1;
        for (std::uint32_t i = 0; i < len; ++i) {
            acc = (k == ScanKind::Sum) ? (T)(acc + ri[i]) : (T)(acc * ri[i]);
            ro[i] = acc;
        }
    }
    return o;
}

// ────────────────────────── normalize (row-local) ────────────────────────
inline std::vector<float> normalize(NormKind k, const std::vector<float>& in, std::uint32_t rows,
                                    std::uint32_t len) {
    std::vector<float> o(in.size());
    for (std::uint32_t r = 0; r < rows; ++r) {
        const float* ri = in.data() + (std::size_t)r * len;
        float* ro = o.data() + (std::size_t)r * len;
        if (k == NormKind::L2Norm) {
            float ss = 0.f;
            for (std::uint32_t i = 0; i < len; ++i) ss += ri[i] * ri[i];
            float inv = 1.f / std::sqrt(ss);
            for (std::uint32_t i = 0; i < len; ++i) ro[i] = ri[i] * inv;
            continue;
        }
        float m = neg_inf();
        for (std::uint32_t i = 0; i < len; ++i) m = std::max(m, ri[i]);
        float sum = 0.f;
        for (std::uint32_t i = 0; i < len; ++i) sum += std::exp(ri[i] - m);
        if (k == NormKind::Softmax) {
            float inv = 1.f / sum;
            for (std::uint32_t i = 0; i < len; ++i) ro[i] = std::exp(ri[i] - m) * inv;
        } else {
            float lse = m + std::log(sum);
            for (std::uint32_t i = 0; i < len; ++i) ro[i] = ri[i] - lse;
        }
    }
    return o;
}

// ──────────────────────────────── sampling ───────────────────────────────
inline std::vector<float> mask_apply(const std::vector<float>& logits,
                                     const std::vector<std::uint8_t>& mask) {
    std::vector<float> o(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) o[i] = mask[i] ? logits[i] : neg_inf();
    return o;
}
inline std::vector<float> gumbel(const std::vector<std::uint32_t>& row_seed, std::uint32_t stream,
                                 std::uint32_t rows, std::uint32_t len) {
    std::vector<float> o((std::size_t)rows * len);
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::uint64_t se = h_seed_eff_stream(row_seed[r], stream);
        for (std::uint32_t j = 0; j < len; ++j) o[(std::size_t)r * len + j] = h_gumbel_noise(se, (int)j);
    }
    return o;
}
// rng (0x70 ambient): per-row draw; gumbel=true → -log(-log(u)), else uniform.
inline std::vector<float> rng_ambient(const std::vector<std::uint32_t>& row_seed, std::uint32_t stream,
                                      std::uint32_t rows, std::uint32_t len, bool gumbel) {
    std::vector<float> o((std::size_t)rows * len);
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::uint64_t se = h_seed_eff_stream(row_seed[r], stream);
        for (std::uint32_t j = 0; j < len; ++j) {
            float u = h_hash_uniform(se, (int)j);
            o[(std::size_t)r * len + j] = gumbel ? -std::log(-std::log(u)) : u;
        }
    }
    return o;
}
// rng_keyed (0x71): seed64 = splitmix64((key<<32)|ctr); element j → hash_uniform.
inline std::vector<float> rng_keyed(std::uint32_t key, std::uint32_t ctr, std::uint64_t numel, bool gumbel) {
    std::uint64_t seed64 = h_splitmix64(((std::uint64_t)key << 32) | (std::uint64_t)ctr);
    std::vector<float> o(numel);
    for (std::uint64_t j = 0; j < numel; ++j) {
        float u = h_hash_uniform(seed64, (int)j);
        o[j] = gumbel ? -std::log(-std::log(u)) : u;
    }
    return o;
}
// mask_apply_packed: bit j (word j>>5, bit j&31), 1 = keep, else -inf. Per row.
inline std::vector<float> mask_apply_packed(const std::vector<float>& logits,
                                            const std::vector<std::uint32_t>& mask,
                                            std::uint32_t rows, std::uint32_t len, std::uint32_t mask_words) {
    std::vector<float> o(logits.size());
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t j = 0; j < len; ++j) {
            std::uint32_t word = mask[(std::size_t)r * mask_words + (j >> 5)];
            bool keep = (word >> (j & 31)) & 1u;
            o[(std::size_t)r * len + j] = keep ? logits[(std::size_t)r * len + j] : neg_inf();
        }
    return o;
}
// sort_desc row-local: descending, ties → lower original index; NaN below −inf.
inline void sort_desc(const std::vector<float>& in, std::uint32_t rows, std::uint32_t len,
                      std::vector<float>& out_val, std::vector<std::uint32_t>& out_idx) {
    out_val.assign(in.size(), 0.f);
    out_idx.assign(in.size(), 0u);
    for (std::uint32_t r = 0; r < rows; ++r) {
        const float* row = in.data() + (std::size_t)r * len;
        std::vector<std::uint32_t> order(len);
        for (std::uint32_t i = 0; i < len; ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
            float va = row[a], vb = row[b];
            bool na = std::isnan(va), nb = std::isnan(vb);
            if (na != nb) return nb;          // NaN sorts last
            if (na && nb) return a < b;
            if (va != vb) return va > vb;      // descending
            return a < b;                      // ties → lower index
        });
        for (std::uint32_t i = 0; i < len; ++i) {
            out_val[(std::size_t)r * len + i] = row[order[i]];
            out_idx[(std::size_t)r * len + i] = order[i];
        }
    }
}

// ────────────────────────────── order family ─────────────────────────────
inline std::vector<std::uint8_t> rank_le(const std::vector<float>& in, std::uint32_t rows,
                                         std::uint32_t len, std::uint32_t k) {
    std::vector<std::uint8_t> o(in.size());
    for (std::uint32_t r = 0; r < rows; ++r) {
        const float* row = in.data() + (std::size_t)r * len;
        for (std::uint32_t i = 0; i < len; ++i) {
            std::uint32_t greater = 0;
            for (std::uint32_t j = 0; j < len; ++j) greater += (row[j] > row[i]) ? 1u : 0u;
            o[(std::size_t)r * len + i] = (greater < k) ? 1u : 0u;
        }
    }
    return o;
}
inline std::vector<float> pivot_threshold_rankle(const std::vector<float>& in, std::uint32_t rows,
                                                 std::uint32_t len, std::uint32_t k) {
    std::vector<float> o(in.size());
    for (std::uint32_t r = 0; r < rows; ++r) {
        const float* row = in.data() + (std::size_t)r * len;
        for (std::uint32_t i = 0; i < len; ++i) {
            std::uint32_t greater = 0;
            for (std::uint32_t j = 0; j < len; ++j) greater += (row[j] > row[i]) ? 1u : 0u;
            o[(std::size_t)r * len + i] = (greater < k) ? row[i] : neg_inf();
        }
    }
    return o;
}

// ─────────────────────── library kernels (top_k, matmul) ─────────────────
inline void topk(const std::vector<float>& in, std::uint32_t rows, std::uint32_t len,
                 std::uint32_t k, std::vector<float>& out_val, std::vector<std::uint32_t>& out_idx) {
    out_val.assign((std::size_t)rows * k, 0.f);
    out_idx.assign((std::size_t)rows * k, 0u);
    for (std::uint32_t r = 0; r < rows; ++r) {
        const float* row = in.data() + (std::size_t)r * len;
        std::vector<std::uint8_t> taken(len, 0);
        for (std::uint32_t p = 0; p < k; ++p) {
            float best = neg_inf();
            std::uint32_t bi = 0;
            for (std::uint32_t i = 0; i < len; ++i) if (!taken[i] && row[i] > best) { best = row[i]; bi = i; }
            out_val[(std::size_t)r * k + p] = best;
            out_idx[(std::size_t)r * k + p] = bi;
            taken[bi] = 1;
        }
    }
}
inline std::vector<float> matmul(const std::vector<float>& A, const std::vector<float>& B,
                                 std::uint32_t M, std::uint32_t K, std::uint32_t N) {
    std::vector<float> C((std::size_t)M * N, 0.f);
    for (std::uint32_t m = 0; m < M; ++m)
        for (std::uint32_t n = 0; n < N; ++n) {
            float acc = 0.f;
            for (std::uint32_t kk = 0; kk < K; ++kk) acc += A[(std::size_t)m * K + kk] * B[(std::size_t)kk * N + n];
            C[(std::size_t)m * N + n] = acc;
        }
    return C;
}
template <class T>
std::vector<T> broadcast(const std::vector<T>& src, std::uint32_t rows, std::uint32_t len, int mode) {
    std::vector<T> o((std::size_t)rows * len);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t j = 0; j < len; ++j)
            o[(std::size_t)r * len + j] = (mode == 0) ? src[0] : src[r];
    return o;
}
template <class T>
std::vector<T> transpose(const std::vector<T>& src, std::uint32_t rows, std::uint32_t cols) {
    std::vector<T> o((std::size_t)rows * cols);
    for (std::uint32_t y = 0; y < rows; ++y)
        for (std::uint32_t x = 0; x < cols; ++x)
            o[(std::size_t)x * rows + y] = src[(std::size_t)y * cols + x];
    return o;
}

}  // namespace pie_cuda_driver::ptir::host_eval
