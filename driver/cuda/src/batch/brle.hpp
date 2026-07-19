#pragma once

// Packed attention-mask decoder.
//
// Pie sends one packed-u32 row per token in `flattened_masks`, indexed by
// `mask_indptr[k]:mask_indptr[k+1]`. Bit 0 of each word is the lowest key
// position; a set bit means "attend to this key position".
//
// For flashinfer's `MaskMode::kCustom` we need a packed bitmap per request
// laid out as `qo_len_r × kv_len_r` bits. Rows shorter than `kv_len_r`
// are padded with zeros.

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

// Detect whether the packed rows encode a pure causal pattern (each token attends
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

// Recognize a causal prefix whose physical page CSR includes reserved tail
// capacity. Returns each request's logical post-write KV length so callers can
// trim those unused pages before taking the standard causal attention path.
bool causal_prefix_lengths(
    std::span<const std::uint32_t> flattened_masks,
    std::span<const std::uint32_t> mask_indptr,
    std::span<const std::uint32_t> qo_indptr_h,
    std::span<const std::uint32_t> kv_page_indptr_h,
    std::span<const std::uint32_t> kv_last_page_lens_h,
    int page_size,
    std::vector<std::uint32_t>& lengths);

// Repack the u32 rows into FlashInfer's concatenated byte bitmap layout.
DecodedMasks decode(
    std::span<const std::uint32_t> flattened_masks,
    std::span<const std::uint32_t> mask_indptr,
    std::span<const std::uint32_t> qo_indptr_h,
    std::span<const std::uint32_t> kv_page_indptr_h,
    std::span<const std::uint32_t> kv_last_page_lens_h,
    int page_size);

}  // namespace pie_cuda_driver::brle
