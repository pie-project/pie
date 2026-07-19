#pragma once
// batch_schedule.hpp — beta's M>1 batch scheduling for the raw-Metal executor.
//
// Pure (Metal-free, no Metal/ObjC, no schema dependency): derives the batch shape from
// the marshaled CSR/SoA view that pie-core hands the driver per fire, and exposes the
// per-request token/KV spans the M>1 dispatch + paged-attention read-walk need.
//
// Batch model (mirrors driver/cuda/src/executor/executor.cpp + model/qwen3_5_forward.cpp,
// grounded by wiki mac-multibatch-gap-analysis):
//   * N = total_tokens = token_ids.len  — the batch/row dim (linear layers become GEMM,
//     M=N; elementwise/norm widen by N rows).
//   * R = num_requests = qo_indptr.len - 1.
//   * Prefill vs decode is DERIVED, not declared: is_pure_decode = (R>0) AND every request
//     has a 1-token qo span (N==R). Any span>1 ⇒ mixed/prefill (CUDA executor.cpp:2335).
//
// At M=1 (single-stream decode) this collapses to N=R=1, is_pure_decode=true, span=[0,1) —
// so the existing GEMV/decode-sdpa fast path is selected unchanged (no churn).
//
// CSR contract (locked w/ delta+alpha, #mac; schema view.hpp:124-135, all u32 except flags):
//   token_ids[N], position_ids[N], qo_indptr[R+1], kv_page_indptr[R+1],
//   kv_page_indices[total_pages], kv_last_page_lens[R], rs_slot_ids[R], rs_slot_flags[R](u8).

#include <cstdint>
#include <string>
#include <vector>

namespace pie::metal {

struct BatchStepInputs {
    std::vector<std::uint32_t> token_ids;
    std::vector<std::uint32_t> position_ids;
    std::vector<std::uint32_t> qo_indptr;
    std::vector<std::uint32_t> kv_page_indptr;
    std::vector<std::uint32_t> kv_page_indices;
    std::vector<std::uint32_t> kv_last_page_lens;
    std::vector<std::uint32_t> rs_slot_ids;
    std::vector<std::uint8_t> rs_slot_flags;
    std::vector<std::uint32_t> w_page;
    std::vector<std::uint32_t> w_off;
    std::vector<std::uint8_t> attention_mask;
    std::vector<std::uint8_t> attention_mask_enabled;
    std::uint32_t attention_mask_stride = 0;
};

// Per-request spans, precomputed once per fire so the dispatch + read-walk avoid re-deriving.
struct RequestSpan {
    uint32_t qo_lo;          // token range [qo_lo, qo_hi) into token_ids/position_ids
    uint32_t qo_hi;
    uint32_t new_tokens;     // qo_hi - qo_lo (1 for a decode row, >1 for prefill)
    uint32_t pages_first;    // kv_page_indices base index for this request
    uint32_t num_pages;      // kv_page_indptr[r+1] - kv_page_indptr[r]
    uint32_t seqlen;         // total KV length AFTER this fire's tokens are appended:
                             //   (num_pages-1)*page_size + kv_last_page_lens[r]
    uint32_t pre_kv_len;     // KV length BEFORE this fire (seqlen - new_tokens) = history
    uint32_t rs_slot;        // recurrent-state slot id (GDN); 0 at M=1
    bool     rs_is_new;      // rs_slot_flags bit: fresh sequence ⇒ zero state before use
};

struct BatchSchedule {
    int  N = 0;                       // total tokens (row/batch dim)
    int  R = 0;                       // number of requests
    bool is_pure_decode = false;      // N==R && every span==1 → decode fast path
    int  page_size = 32;              // runtime kv_page_size (default 32)
    std::vector<RequestSpan> spans;        // size R
    std::vector<uint32_t>    req_of_token; // size N: owning request r per token (== bind::ReqOfToken;
                                           // alpha's entry marshals this straight into IO::ReqOfToken)
    std::vector<uint32_t>    slot_of_token; // size N: per-token GDN recurrent-state slot =
                                           // rs_slot_ids[req_of_token[t]]. This is what gdn_core_slotted
                                           // binds (it indexes slot_ids[b_idx] by TOKEN row [N], not the
                                           // raw per-request [R] RsSlotIds). For pure-decode (N==R) it
                                           // equals RsSlotIds[R] elementwise (alpha may bind RsSlotIds
                                           // directly); for mixed/prefill (N>R) it's the derived expansion.

    bool m1() const { return N == 1 && R == 1; }  // the shipped single-stream fast path
};

// Build the schedule from the marshaled CSR arrays. All pointers are borrowed (valid for the
// fire); `*_flags` may be null (⇒ rs_is_new=false). page_size = runtime kv_page_size.
// Pure + allocation-only — no Metal. Unit-testable standalone.
inline BatchSchedule build_batch_schedule(
    const uint32_t* token_ids, int token_ids_len,
    const uint32_t* qo_indptr,            // [R+1]
    const uint32_t* kv_page_indptr,       // [R+1]
    const uint32_t* kv_last_page_lens,    // [R]
    const uint32_t* rs_slot_ids,          // [R] or null
    const uint8_t*  rs_slot_flags,        // [R] or null
    int qo_indptr_len,                    // == R+1
    int page_size)
{
    BatchSchedule s;
    s.N         = token_ids_len;
    s.R         = qo_indptr_len > 0 ? qo_indptr_len - 1 : 0;
    s.page_size = page_size > 0 ? page_size : 32;

    s.spans.resize(s.R);
    bool pure = (s.R > 0);
    for (int r = 0; r < s.R; ++r) {
        RequestSpan& sp = s.spans[r];
        sp.qo_lo       = qo_indptr[r];
        sp.qo_hi       = qo_indptr[r + 1];
        sp.new_tokens  = sp.qo_hi - sp.qo_lo;
        sp.pages_first = kv_page_indptr[r];
        sp.num_pages   = kv_page_indptr[r + 1] - kv_page_indptr[r];
        sp.seqlen      = sp.num_pages > 0
                       ? (sp.num_pages - 1) * uint32_t(s.page_size) + kv_last_page_lens[r]
                       : 0;
        sp.pre_kv_len  = sp.seqlen - sp.new_tokens;
        sp.rs_slot     = rs_slot_ids ? rs_slot_ids[r] : 0;
        sp.rs_is_new   = rs_slot_flags ? (rs_slot_flags[r] != 0) : false;
        if (sp.new_tokens != 1) pure = false;
    }
    s.is_pure_decode = pure;

    // req_of_token: owning request per token (CUDA find_request equivalent; spans are
    // contiguous + ascending so this is O(N)). Populated into IO::ReqOfToken by alpha's entry.
    s.req_of_token.resize(s.N, 0);
    for (int r = 0, t = 0; r < s.R; ++r)
        for (uint32_t e = s.spans[r].qo_hi; t < int(e) && t < s.N; ++t)
            s.req_of_token[t] = uint32_t(r);

    // slot_of_token: per-token GDN slot expansion = rs_slot_ids[req_of_token[t]]. gdn_core_slotted
    // reads this by token row [N]. Equals RsSlotIds[R] elementwise when pure-decode (N==R).
    s.slot_of_token.resize(s.N, 0);
    for (int t = 0; t < s.N; ++t)
        s.slot_of_token[t] = s.spans[s.req_of_token[t]].rs_slot;

    return s;
}

// find_request(qo_indptr, R, t): owning request r with qo_indptr[r] <= t < qo_indptr[r+1].
// Linear (spans are few + contiguous); mirrors CUDA kv_paged.cu find_request. For host-side
// grid derivation; the GPU kernels carry tok_req as a bound buffer (no per-thread search).
inline int find_request(const uint32_t* qo_indptr, int R, int t) {
    for (int r = 0; r < R; ++r)
        if (uint32_t(t) >= qo_indptr[r] && uint32_t(t) < qo_indptr[r + 1]) return r;
    return R > 0 ? R - 1 : 0;
}

// Final host-side safety gate before a paged GPU dispatch.  It validates the
// exact address formula used by kv_append_paged/sdpa_paged, including explicit
// writes, so invalid geometry is rejected before any pool cell can be touched.
inline bool validate_paged_batch(const BatchSchedule& s,
                                 const std::vector<uint32_t>& position_ids,
                                 const std::vector<uint32_t>& page_indices,
                                 const std::vector<uint32_t>& w_page,
                                 const std::vector<uint32_t>& w_off,
                                 uint32_t total_pages, uint32_t max_slots,
                                 std::string* err = nullptr) {
    auto fail = [&](const char* why) {
        if (err) *err = why;
        return false;
    };
    if (s.N <= 0 || s.R <= 0 || int(position_ids.size()) != s.N ||
        int(w_page.size()) != s.N || int(w_off.size()) != s.N ||
        s.spans.size() != size_t(s.R) || s.req_of_token.size() != size_t(s.N) ||
        s.slot_of_token.size() != size_t(s.N) || s.page_size <= 0)
        return fail("malformed paged batch vector sizes");
    uint32_t expected_qo = 0, expected_pages = 0;
    for (int r = 0; r < s.R; ++r) {
        const RequestSpan& sp = s.spans[size_t(r)];
        if (sp.qo_lo != expected_qo || sp.qo_lo >= sp.qo_hi || sp.qo_hi > uint32_t(s.N) ||
            sp.pages_first != expected_pages || sp.num_pages == 0 ||
            sp.pages_first + sp.num_pages > page_indices.size() ||
            sp.rs_slot >= max_slots || sp.seqlen == 0 || sp.new_tokens > sp.seqlen ||
            sp.pre_kv_len > sp.seqlen)
            return fail("malformed request CSR span");
        const uint32_t last = sp.seqlen - (sp.num_pages - 1) * uint32_t(s.page_size);
        if (last == 0 || last > uint32_t(s.page_size))
            return fail("invalid final page length");
        for (uint32_t j = 0; j < sp.num_pages; ++j)
            if (page_indices[sp.pages_first + j] >= total_pages)
                return fail("physical page id exceeds pool");
        expected_qo = sp.qo_hi;
        expected_pages = sp.pages_first + sp.num_pages;
    }
    if (expected_qo != uint32_t(s.N) || expected_pages != page_indices.size())
        return fail("CSR does not cover exactly the batch token/page arrays");
    for (int t = 0; t < s.N; ++t) {
        const uint32_t r = s.req_of_token[size_t(t)];
        if (r >= uint32_t(s.R) || s.slot_of_token[size_t(t)] != s.spans[r].rs_slot)
            return fail("invalid request/slot expansion");
        const RequestSpan& sp = s.spans[r];
        const uint32_t pos = position_ids[size_t(t)];
        if (pos >= sp.seqlen || w_page[size_t(t)] >= total_pages ||
            w_off[size_t(t)] >= uint32_t(s.page_size))
            return fail("position or write descriptor exceeds extent");
        const uint32_t expected = page_indices[sp.pages_first + pos / uint32_t(s.page_size)];
        if (w_page[size_t(t)] != expected || w_off[size_t(t)] != pos % uint32_t(s.page_size))
            return fail("write descriptor disagrees with page CSR");
    }
    return true;
}

inline bool validate_paged_batch_capacity(const BatchSchedule& s, uint32_t max_tokens,
                                          uint32_t max_requests, std::string* err = nullptr) {
    if (s.N > 0 && uint32_t(s.N) <= max_tokens && s.R > 0 && uint32_t(s.R) <= max_requests)
        return true;
    if (err) *err = "paged batch exceeds configured token/request capacity";
    return false;
}

}  // namespace pie::metal
