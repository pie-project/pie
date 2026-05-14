#include "brle.hpp"

#include <algorithm>
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

}  // namespace

bool is_pure_causal(
    std::span<const std::uint32_t> flattened_masks,
    std::span<const std::uint32_t> mask_indptr,
    std::span<const std::uint32_t> qo_indptr_h,
    std::span<const std::uint32_t> kv_page_indptr_h,
    std::span<const std::uint32_t> kv_last_page_lens_h,
    int page_size)
{
    const int R = static_cast<int>(qo_indptr_h.size()) - 1;
    if (R <= 0) return true;

    for (int r = 0; r < R; ++r) {
        const int kv_len = kv_len_for(r, page_size,
                                      kv_page_indptr_h, kv_last_page_lens_h);
        const int qo_lo = static_cast<int>(qo_indptr_h[r]);
        const int qo_hi = static_cast<int>(qo_indptr_h[r + 1]);
        const int qo_len = qo_hi - qo_lo;
        const int pre_kv = kv_len - qo_len;

        for (int q = 0; q < qo_len; ++q) {
            const int token_idx = qo_lo + q;
            const int p = pre_kv + q;  // absolute KV position for this token
            const int rle_start = static_cast<int>(mask_indptr[token_idx]);
            const int rle_end   = static_cast<int>(mask_indptr[token_idx + 1]);

            // Causal pattern: BRLE = [0, p+1] (one zero-run of 0, one
            // one-run of length p+1). Anything else means custom mask.
            if (rle_end - rle_start != 2) return false;
            if (flattened_masks[rle_start] != 0u) return false;
            if (flattened_masks[rle_start + 1] != static_cast<std::uint32_t>(p + 1))
                return false;
        }
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

    // Second pass: decode BRLE → packed bits.
    for (int r = 0; r < R; ++r) {
        const int kv_len = kv_len_for(r, page_size,
                                      kv_page_indptr_h, kv_last_page_lens_h);
        const int qo_lo = static_cast<int>(qo_indptr_h[r]);
        const int qo_hi = static_cast<int>(qo_indptr_h[r + 1]);
        const int qo_len = qo_hi - qo_lo;
        std::uint8_t* row_base = out.packed.data() + out.mask_indptr[r];

        for (int q = 0; q < qo_len; ++q) {
            const int token_idx = qo_lo + q;
            const int rle_start = static_cast<int>(mask_indptr[token_idx]);
            const int rle_end   = static_cast<int>(mask_indptr[token_idx + 1]);

            // BRLE always starts with a (possibly empty) false run, then
            // alternates true / false / true / ...
            int kv_pos = 0;
            bool is_true_run = false;

            for (int run_idx = rle_start; run_idx < rle_end; ++run_idx) {
                int run_len = static_cast<int>(flattened_masks[run_idx]);
                if (kv_pos >= kv_len) break;
                int eff_len = std::min(run_len, kv_len - kv_pos);

                if (is_true_run && eff_len > 0) {
                    // Set `eff_len` consecutive bits starting at row q,
                    // column kv_pos. Output offset = q * kv_len + kv_pos.
                    const std::int64_t row_bit_off =
                        static_cast<std::int64_t>(q) *
                        static_cast<std::int64_t>(kv_len);
                    for (int i = 0; i < eff_len; ++i) {
                        const std::int64_t bit = row_bit_off + kv_pos + i;
                        row_base[bit / 8] |=
                            static_cast<std::uint8_t>(1u << (bit % 8));
                    }
                }
                kv_pos += eff_len;
                is_true_run = !is_true_run;
            }
            // Bits past the BRLE-encoded `kv_pos` stay 0 (false), which is
            // the correct interpretation of "implicitly masked".
        }
    }

    return out;
}

}  // namespace pie_cuda_driver::brle
