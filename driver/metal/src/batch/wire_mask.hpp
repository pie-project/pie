#pragma once

// What a wire attention mask says, when it says nothing this driver's
// attention does not already do.
//
// Pie hands the driver one packed-u32 row per query token: bit `k` of row `q`
// set means "row q attends key position k", rows delimited by `word_indptr`.
// Metal's paged attention takes a DENSE mask (one byte a lane, see
// `sdpa_paged.metal`'s `attention_mask`), and nothing here builds one from the
// wire form -- so a launch carrying wire masks used to be refused outright.
//
// Most of them do not need building. A prefill's mask is the causal pattern:
// row q attends exactly key positions `[0, q]`, and the last row reaches the
// request's whole KV length. That is the bound `sdpa_paged` applies on its
// own, from `qo_indptr` and the page CSR, whether or not a mask is bound. So a
// mask that says only this can be dropped and the answer is bit-identical --
// not approximated, IDENTICAL, because the kernel's own predicate and the
// mask's are then the same predicate.
//
// Anything else -- a window, a sink, a prefix the router cut short, a mask
// that skips a key in the middle -- is a claim the kernel would not make by
// itself, and this returns false so the caller refuses rather than quietly
// attending to keys the mask excluded. CUDA reaches the same conclusion from
// the same bytes in `driver/cuda/src/batch/brle.cpp`; this is that predicate,
// narrowed to the one question Metal needs answered.

#include <bit>
#include <cstdint>
#include <vector>

#include "pie_driver_abi.h"

namespace pie::metal::wire_mask {

namespace {

inline bool row_is_prefix(
    const PieU32Slice& words,
    std::uint32_t begin,
    std::uint32_t end,
    std::uint32_t& prefix) {
    prefix = 0;
    bool saw_tail = false;
    for (std::uint32_t i = begin; i < end; ++i) {
        const std::uint32_t word = words.ptr[i];
        if (saw_tail) {
            // Past the run of ones every remaining bit must be clear, or the
            // row attends something its prefix does not cover.
            if (word != 0) return false;
            continue;
        }
        const std::uint32_t ones = static_cast<std::uint32_t>(
            std::countr_one(word));
        prefix += ones;
        if (ones != 32) {
            if ((word >> ones) != 0) return false;
            saw_tail = true;
        }
    }
    return prefix != 0;
}


}  // namespace

/// The per-request logical KV length each request's mask declares, when every
/// row of it is a causal prefix. False when any row is not.
///
/// Separate from "the mask asks for exactly what the CSR says" because the two
/// differ in practice and the difference is the whole point: a prefill's page
/// CSR carries pages RESERVED for the decode that follows, so a request whose
/// mask stops at key 53 can arrive with four pages and a nominal length of
/// 117. The reserved tail is not history, and attending to it reads whatever
/// those pages held before. The mask's prefix is the length that is true, and
/// a caller that trims the CSR to it gets exactly the causal attention the
/// mask asked for.
inline bool causal_prefix_lengths(
    const PieMaskWordsDesc& masks,
    const PieU32Slice& qo_indptr,
    std::vector<std::uint32_t>& lengths) {
    lengths.clear();
    if (qo_indptr.len < 2) return false;
    const std::size_t requests = qo_indptr.len - 1;
    const std::uint32_t rows = qo_indptr.ptr[requests];
    if (masks.word_indptr.len < static_cast<std::size_t>(rows) + 1) return false;

    lengths.reserve(requests);
    for (std::size_t r = 0; r < requests; ++r) {
        const std::uint32_t lo = qo_indptr.ptr[r];
        const std::uint32_t hi = qo_indptr.ptr[r + 1];
        if (hi <= lo) return false;
        std::uint32_t previous = 0;
        for (std::uint32_t q = lo; q < hi; ++q) {
            const std::uint32_t begin = masks.word_indptr.ptr[q];
            const std::uint32_t end = masks.word_indptr.ptr[q + 1];
            if (begin > end || end > masks.words.len) return false;
            std::uint32_t prefix = 0;
            if (!row_is_prefix(masks.words, begin, end, prefix)) return false;
            // Each row reaches exactly one key further than the row above it.
            // A step of anything else is a pattern with a shape of its own.
            if (q != lo && prefix != previous + 1) return false;
            previous = prefix;
        }
        lengths.push_back(previous);
    }
    return true;
}

/// Cut a page CSR back to the lengths a causal mask vouches for.
///
/// `lengths[r]` keys survive for request `r`; the pages past them were
/// reserved, not written. Keeps each request's first `ceil(len / page_size)`
/// page ids and restates its last-page fill, so the geometry the kernel reads
/// describes only history.
inline void trim_kv_to_lengths(
    std::vector<std::uint32_t>& kv_page_indices,
    std::vector<std::uint32_t>& kv_page_indptr,
    std::vector<std::uint32_t>& kv_last_page_lens,
    const std::vector<std::uint32_t>& lengths,
    std::uint32_t page_size) {
    if (page_size == 0 || kv_page_indptr.size() != lengths.size() + 1) return;
    std::vector<std::uint32_t> pages;
    std::vector<std::uint32_t> indptr;
    pages.reserve(kv_page_indices.size());
    indptr.reserve(kv_page_indptr.size());
    indptr.push_back(0);
    for (std::size_t r = 0; r < lengths.size(); ++r) {
        const std::uint32_t have = kv_page_indptr[r + 1] - kv_page_indptr[r];
        const std::uint32_t want = (lengths[r] + page_size - 1) / page_size;
        const std::uint32_t keep = want < have ? want : have;
        for (std::uint32_t p = 0; p < keep; ++p) {
            pages.push_back(kv_page_indices[kv_page_indptr[r] + p]);
        }
        indptr.push_back(static_cast<std::uint32_t>(pages.size()));
        kv_last_page_lens[r] =
            keep == 0 ? 0 : lengths[r] - (keep - 1) * page_size;
    }
    kv_page_indices.swap(pages);
    kv_page_indptr.swap(indptr);
}

}  // namespace pie::metal::wire_mask
