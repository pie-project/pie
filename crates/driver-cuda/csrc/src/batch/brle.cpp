#include "batch/brle.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>

namespace pie_cuda_driver::brle {

namespace {

// Per-request KV length: (num_pages - 1) * page_size + last_page_len.
// Returns the *post-write* length (matches wire semantics).
int kv_len_for(
    int r, int page_size,
    std::span<const std::uint32_t> kv_page_indptr_h,
    std::span<const std::uint32_t> kv_last_page_lens_h)
{
    const int num_pages =
        static_cast<int>(kv_page_indptr_h[r + 1] - kv_page_indptr_h[r]);
    if (num_pages <= 0) return 0;
    return (num_pages - 1) * page_size +
           static_cast<int>(kv_last_page_lens_h[r]);
}

bool has_valid_layout(
    std::span<const std::uint32_t> flattened_masks,
    std::span<const std::uint32_t> mask_indptr,
    std::span<const std::uint32_t> qo_indptr_h,
    std::span<const std::uint32_t> kv_page_indptr_h,
    std::span<const std::uint32_t> kv_last_page_lens_h)
{
    if (qo_indptr_h.empty()) return false;
    const int R = static_cast<int>(qo_indptr_h.size()) - 1;
    if (R <= 0) return true;
    if (kv_page_indptr_h.size() < static_cast<std::size_t>(R + 1))
        return false;
    if (kv_last_page_lens_h.size() < static_cast<std::size_t>(R)) return false;

    const std::uint32_t total_rows = qo_indptr_h.back();
    if (mask_indptr.size() < static_cast<std::size_t>(total_rows) + 1)
        return false;

    for (int r = 0; r < R; ++r) {
        if (qo_indptr_h[r] > qo_indptr_h[r + 1]) return false;
        if (kv_page_indptr_h[r] > kv_page_indptr_h[r + 1]) return false;
    }
    for (std::size_t row = 0; row < static_cast<std::size_t>(total_rows);
         ++row) {
        if (mask_indptr[row] > mask_indptr[row + 1]) return false;
        if (mask_indptr[row + 1] > flattened_masks.size()) return false;
    }
    return true;
}

void require_valid_layout(
    std::span<const std::uint32_t> flattened_masks,
    std::span<const std::uint32_t> mask_indptr,
    std::span<const std::uint32_t> qo_indptr_h,
    std::span<const std::uint32_t> kv_page_indptr_h,
    std::span<const std::uint32_t> kv_last_page_lens_h)
{
    if (!has_valid_layout(flattened_masks, mask_indptr, qo_indptr_h,
                          kv_page_indptr_h, kv_last_page_lens_h)) {
        throw std::runtime_error("brle::decode: malformed mask layout");
    }
}

bool packed_bit(
    std::span<const std::uint32_t> words,
    std::uint32_t begin,
    std::uint32_t end,
    std::uint32_t bit) {
    const std::uint32_t word = bit / 32;
    if (begin + word >= end) return false;
    return ((words[begin + word] >> (bit % 32)) & 1u) != 0;
}

bool leading_true_prefix(
    std::span<const std::uint32_t> words,
    std::uint32_t begin,
    std::uint32_t end,
    std::uint32_t& prefix) {
    prefix = 0;
    bool saw_tail = false;
    for (std::uint32_t index = begin; index < end; ++index) {
        const std::uint32_t word = words[index];
        if (saw_tail) {
            if (word != 0) return false;
            continue;
        }
        const std::uint32_t ones = std::countr_one(word);
        prefix += ones;
        if (ones != 32) {
            if ((word >> ones) != 0) return false;
            saw_tail = true;
        }
    }
    return prefix != 0;
}

}  // namespace

bool is_pure_causal(
    std::span<const std::uint32_t> flattened_masks,
    std::span<const std::uint32_t> mask_indptr,
    std::span<const std::uint32_t> qo_indptr_h,
    std::span<const std::uint32_t> kv_page_indptr_h,
    std::span<const std::uint32_t> kv_last_page_lens_h,
    int page_size)
{
    std::vector<std::uint32_t> lengths;
    if (!causal_prefix_lengths(
            flattened_masks, mask_indptr, qo_indptr_h,
            kv_page_indptr_h, kv_last_page_lens_h,
            page_size, lengths)) {
        return false;
    }
    const int R = static_cast<int>(qo_indptr_h.size()) - 1;
    for (int r = 0; r < R; ++r) {
        if (lengths[r] != static_cast<std::uint32_t>(kv_len_for(
                r, page_size, kv_page_indptr_h,
                kv_last_page_lens_h))) {
            return false;
        }
    }
    return true;
}

bool causal_prefix_lengths(
    std::span<const std::uint32_t> flattened_masks,
    std::span<const std::uint32_t> mask_indptr,
    std::span<const std::uint32_t> qo_indptr_h,
    std::span<const std::uint32_t> kv_page_indptr_h,
    std::span<const std::uint32_t> kv_last_page_lens_h,
    int page_size,
    std::vector<std::uint32_t>& lengths)
{
    lengths.clear();
    const int R = static_cast<int>(qo_indptr_h.size()) - 1;
    if (R <= 0) return true;
    if (!has_valid_layout(flattened_masks, mask_indptr, qo_indptr_h,
                          kv_page_indptr_h, kv_last_page_lens_h)) {
        return false;
    }

    lengths.reserve(R);
    for (int r = 0; r < R; ++r) {
        const int physical_kv_len = kv_len_for(
            r, page_size, kv_page_indptr_h, kv_last_page_lens_h);
        const int qo_lo = static_cast<int>(qo_indptr_h[r]);
        const int qo_hi = static_cast<int>(qo_indptr_h[r + 1]);
        const int qo_len = qo_hi - qo_lo;
        if (qo_len <= 0) return false;
        std::uint32_t previous_prefix = 0;

        for (int q = 0; q < qo_len; ++q) {
            const int token_idx = qo_lo + q;
            std::uint32_t prefix = 0;
            if (!leading_true_prefix(
                    flattened_masks,
                    mask_indptr[token_idx],
                    mask_indptr[token_idx + 1],
                    prefix) ||
                (q != 0 && prefix != previous_prefix + 1)) {
                return false;
            }
            previous_prefix = prefix;
        }
        if (previous_prefix > static_cast<std::uint32_t>(physical_kv_len)) {
            return false;
        }
        lengths.push_back(previous_prefix);
    }
    return true;
}

DecodedMasks decode(
    std::span<const std::uint32_t> flattened_masks,
    std::span<const std::uint32_t> mask_indptr,
    std::span<const std::uint32_t> qo_indptr_h,
    std::span<const std::uint32_t> kv_page_indptr_h,
    std::span<const std::uint32_t> kv_last_page_lens_h,
    int page_size)
{
    DecodedMasks out;
    const int R = static_cast<int>(qo_indptr_h.size()) - 1;
    if (R <= 0) {
        out.mask_indptr = {0};
        return out;
    }
    require_valid_layout(flattened_masks, mask_indptr, qo_indptr_h,
                         kv_page_indptr_h, kv_last_page_lens_h);

    out.mask_indptr.resize(R + 1);
    out.mask_indptr[0] = 0;

    // First pass: per-request byte counts so we can size `packed`.
    for (int r = 0; r < R; ++r) {
        const int kv_len = kv_len_for(r, page_size,
                                      kv_page_indptr_h, kv_last_page_lens_h);
        const int qo_len =
            static_cast<int>(qo_indptr_h[r + 1] - qo_indptr_h[r]);
        const std::int64_t bits = static_cast<std::int64_t>(qo_len) *
                                  static_cast<std::int64_t>(kv_len);
        const std::int64_t bytes = (bits + 7) / 8;
        out.mask_indptr[r + 1] = out.mask_indptr[r] +
                                 static_cast<std::int32_t>(bytes);
    }
    out.packed.assign(static_cast<std::size_t>(out.mask_indptr.back()), 0);

    // Second pass: repack the row-aligned u32 words into byte-packed requests.
    for (int r = 0; r < R; ++r) {
        const int kv_len = kv_len_for(r, page_size,
                                      kv_page_indptr_h, kv_last_page_lens_h);
        const int qo_lo = static_cast<int>(qo_indptr_h[r]);
        const int qo_hi = static_cast<int>(qo_indptr_h[r + 1]);
        const int qo_len = qo_hi - qo_lo;
        std::uint8_t* row_base = out.packed.data() + out.mask_indptr[r];

        for (int q = 0; q < qo_len; ++q) {
            const int token_idx = qo_lo + q;
            const std::uint32_t word_begin = mask_indptr[token_idx];
            const std::uint32_t word_end = mask_indptr[token_idx + 1];
            const std::int64_t row_bit_off =
                static_cast<std::int64_t>(q) * kv_len;
            for (int key = 0; key < kv_len; ++key) {
                if (!packed_bit(
                        flattened_masks, word_begin, word_end,
                        static_cast<std::uint32_t>(key))) {
                    continue;
                }
                const std::int64_t bit = row_bit_off + key;
                row_base[bit / 8] |=
                    static_cast<std::uint8_t>(1u << (bit % 8));
            }
        }
    }

    return out;
}

}  // namespace pie_cuda_driver::brle
