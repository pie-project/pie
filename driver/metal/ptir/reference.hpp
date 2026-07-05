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

// mask_apply_packed (matrix): per-row packed bitmap, row stride words_per_row.
inline std::vector<float> mask_apply_packed_matrix(const std::vector<float>& logits,
                                                   const std::vector<std::uint32_t>& mask,
                                                   std::uint32_t rows,
                                                   std::uint32_t vocab,
                                                   std::uint32_t words_per_row) {
    std::vector<float> out(logits.size());
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t c = 0; c < vocab; ++c) {
            std::size_t idx = static_cast<std::size_t>(r) * vocab + c;
            std::size_t widx = static_cast<std::size_t>(r) * words_per_row + (c >> 5);
            std::uint32_t word = widx < mask.size() ? mask[widx] : 0u;
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

}  // namespace ptir_metal::ref
