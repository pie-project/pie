#pragma once
//
// CPU reference implementations of the PTIR sampling-IR foundational ops —
// bit-exact ports of interface/sampling-ir/src/eval.rs. Used as the interim
// cross-backend oracle until echo's golden vector files are wired in; the
// Metal kernel output is compared byte-for-byte against these.
//
// Pure C++17, no Metal / MLX dependency.

#include <cstdint>
#include <limits>
#include <vector>

namespace ptir_metal::ref {

inline constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// mask_apply_packed (vector): out[j] = bit_j ? logits[j] : -inf,
// bit_j = (mask[j>>5] >> (j&31)) & 1.
inline std::vector<float> mask_apply_packed(const std::vector<float>& logits,
                                            const std::vector<std::uint32_t>& mask) {
    std::vector<float> out(logits.size());
    for (std::size_t j = 0; j < logits.size(); ++j) {
        std::uint32_t word = (j >> 5) < mask.size() ? mask[j >> 5] : 0u;
        std::uint32_t bit = (word >> (j & 31u)) & 1u;
        out[j] = bit == 1u ? logits[j] : kNegInf;
    }
    return out;
}

// mask_apply_packed (matrix): a SINGLE packed word-row [ceil(vocab/32)]
// broadcast across ALL rows (pinned PTIR contract) — bit index = column, same
// words for every row.
inline std::vector<float> mask_apply_packed_matrix(const std::vector<float>& logits,
                                                   const std::vector<std::uint32_t>& mask,
                                                   std::uint32_t rows,
                                                   std::uint32_t vocab) {
    std::vector<float> out(logits.size());
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t c = 0; c < vocab; ++c) {
            std::size_t idx = static_cast<std::size_t>(r) * vocab + c;
            std::uint32_t word = (c >> 5) < mask.size() ? mask[c >> 5] : 0u;
            std::uint32_t bit = (word >> (c & 31u)) & 1u;
            out[idx] = bit == 1u ? logits[idx] : kNegInf;
        }
    }
    return out;
}

// dselect / Op::Select (F32 arm): out[i] = cond[i] ? a[i] : b[i], with
// per-operand length-1 broadcast (len == 1 => index 0).
inline std::vector<float> dselect_f32(const std::vector<std::uint8_t>& cond,
                                      const std::vector<float>& a,
                                      const std::vector<float>& b) {
    std::size_t n = cond.size();
    if (a.size() > n) n = a.size();
    if (b.size() > n) n = b.size();
    auto pick = [](std::size_t len, std::size_t i) { return len == 1 ? std::size_t(0) : i; };
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        bool c = cond[pick(cond.size(), i)] != 0;
        out[i] = c ? a[pick(a.size(), i)] : b[pick(b.size(), i)];
    }
    return out;
}

// broadcast_matrix / Op::Broadcast (rank<=2, left-aligned row-major).
inline std::vector<float> broadcast_matrix_f32(const std::vector<float>& src,
                                               std::uint32_t src_rows,
                                               std::uint32_t src_cols,
                                               std::uint32_t dst_rows,
                                               std::uint32_t dst_cols) {
    std::vector<float> out(static_cast<std::size_t>(dst_rows) * dst_cols);
    for (std::uint32_t r = 0; r < dst_rows; ++r) {
        for (std::uint32_t c = 0; c < dst_cols; ++c) {
            std::uint32_t sr = src_rows == 1u ? 0u : r;
            std::uint32_t sc = src_cols == 1u ? 0u : c;
            out[static_cast<std::size_t>(r) * dst_cols + c] =
                src[static_cast<std::size_t>(sr) * src_cols + sc];
        }
    }
    return out;
}

// ── elementwise + reductions + scans (bit-exact ports of eval.rs) ────────────

inline std::vector<float> neg_f32(const std::vector<float>& a) {
    std::vector<float> out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) out[i] = -a[i];
    return out;
}

// Binary map with per-operand len-1 broadcast (eval.rs zip_f32).
template <class F>
inline std::vector<float> binary_f32(const std::vector<float>& a,
                                     const std::vector<float>& b, F f) {
    std::size_t n = a.size() > b.size() ? a.size() : b.size();
    auto pick = [](std::size_t l, std::size_t i) { return l == 1 ? std::size_t(0) : i; };
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = f(a[pick(a.size(), i)], b[pick(b.size(), i)]);
    return out;
}

// Comparison → bool bytes (eval.rs cmp).
template <class F>
inline std::vector<std::uint8_t> cmp_f32(const std::vector<float>& a,
                                         const std::vector<float>& b, F f) {
    std::size_t n = a.size() > b.size() ? a.size() : b.size();
    auto pick = [](std::size_t l, std::size_t i) { return l == 1 ? std::size_t(0) : i; };
    std::vector<std::uint8_t> out(n);
    for (std::size_t i = 0; i < n; ++i)
        out[i] = f(a[pick(a.size(), i)], b[pick(b.size(), i)]) ? 1 : 0;
    return out;
}

// Per-row reduce (eval.rs reduce_rows / argmax_rows): sequential fold order.
template <class F>
inline std::vector<float> reduce_rows(const std::vector<float>& in, std::uint32_t rows,
                                      std::uint32_t len, float init, F f) {
    std::vector<float> out(rows);
    for (std::uint32_t r = 0; r < rows; ++r) {
        float acc = init;
        for (std::uint32_t j = 0; j < len; ++j) acc = f(acc, in[r * len + j]);
        out[r] = acc;
    }
    return out;
}

inline std::vector<std::int32_t> argmax_rows(const std::vector<float>& in,
                                             std::uint32_t rows, std::uint32_t len) {
    std::vector<std::int32_t> out(rows);
    for (std::uint32_t r = 0; r < rows; ++r) {
        float best = kNegInf;
        std::int32_t bi = 0;
        for (std::uint32_t j = 0; j < len; ++j) {
            float x = in[r * len + j];
            if (x > best) { best = x; bi = static_cast<std::int32_t>(j); }
        }
        out[r] = bi;
    }
    return out;
}

template <class F>
inline std::vector<float> scan_rows(const std::vector<float>& in, std::uint32_t rows,
                                    std::uint32_t len, float init, F f) {
    std::vector<float> out(in.size());
    for (std::uint32_t r = 0; r < rows; ++r) {
        float acc = init;
        for (std::uint32_t j = 0; j < len; ++j) { acc = f(acc, in[r * len + j]); out[r * len + j] = acc; }
    }
    return out;
}

}  // namespace ptir_metal::ref