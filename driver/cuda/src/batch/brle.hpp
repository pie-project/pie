#pragma once

// BRLE (Bit Run Length Encoding) decoder for attention masks.
//
// Pie sends one BRLE row per token in `flattened_masks`, indexed by
// `mask_indptr[k]:mask_indptr[k+1]`. Within a row the runs alternate
// false, true, false, true, … starting with a (possibly zero-length)
// false run. A `1` bit means "attend to this key position".
//
// For flashinfer's `MaskMode::kCustom` we need a packed bitmap per request
// laid out as `qo_len_r × kv_len_r` bits. We pad rows shorter than
// `kv_len_r` with zeros (consistent with the BRLE convention that bits
// past the row's `valid_len` are implicitly false).

#include <cstdint>
#include <span>
#include <vector>

namespace pie_cuda_driver::brle {

// Output of the host-side decoder.
struct DecodedMasks {
    std::vector<std::uint8_t> packed;       // concatenated per-request bitmaps
    std::vector<std::int32_t> mask_indptr;  // [num_requests + 1], byte offsets
    bool pure_causal = false;               // shortcut hint
};

// Detect whether the BRLE encodes a pure causal pattern (each token attends
// to exactly its prefix of the request's KV history). Cheap O(num_tokens).
//
// `qo_indptr_h`, `kv_page_indptr_h`, `kv_last_page_lens_h` describe the
// request layout (host-side, length R+1 / R+1 / R).
bool is_pure_causal(
    std::span<const std::uint32_t> flattened_masks,
    std::span<const std::uint32_t> mask_indptr,
    std::span<const std::uint32_t> qo_indptr_h,
    std::span<const std::uint32_t> kv_page_indptr_h,
    std::span<const std::uint32_t> kv_last_page_lens_h,
    int page_size);

// Decode the BRLE into the flashinfer-compatible packed bitmap layout.
DecodedMasks decode(
    std::span<const std::uint32_t> flattened_masks,
    std::span<const std::uint32_t> mask_indptr,
    std::span<const std::uint32_t> qo_indptr_h,
    std::span<const std::uint32_t> kv_page_indptr_h,
    std::span<const std::uint32_t> kv_last_page_lens_h,
    int page_size);

}  // namespace pie_cuda_driver::brle
