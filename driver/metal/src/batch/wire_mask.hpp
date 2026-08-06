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

/// The first request whose mask length disagrees with what the page CSR says
/// its KV length is, or -1 when they all agree.
///
/// These two are the same number stated twice, and the inferlet that states
/// them says so: "the page CSR is the SOURCE OF TRUTH for kv_len on the wire
/// ... declaring the whole pool here would claim a kv length the pass does not
/// have and silently corrupt attention". Silently is the problem. When they
/// disagree the mask is a second opinion the driver cannot reconcile -- it
/// cannot know which one the KV WRITE used -- so the only safe answer is to
/// say which two numbers differ and stop.
///
/// The disagreement is not hypothetical and its usual cause is dull: an
/// inferlet paging at one size against a driver configured for another. A
/// 53-token prefill from a `PAGE_T = 16` inferlet arrives as four pages, and a
/// driver reading them at 32 makes that 117 keys against the mask's 53.
inline int first_kv_len_disagreement(
    const std::vector<std::uint32_t>& lengths,
    const PieU32Slice& kv_page_indptr,
    const PieU32Slice& kv_last_page_lens,
    std::uint32_t page_size,
    std::uint32_t& claimed) {
    for (std::size_t r = 0; r < lengths.size(); ++r) {
        const std::uint32_t pages = kv_page_indptr.ptr[r + 1] - kv_page_indptr.ptr[r];
        const std::uint32_t kv =
            pages == 0 ? 0u : (pages - 1) * page_size + kv_last_page_lens.ptr[r];
        if (kv != lengths[r]) {
            claimed = kv;
            return static_cast<int>(r);
        }
    }
    return -1;
}

}  // namespace pie::metal::wire_mask
