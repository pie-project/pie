#pragma once
// forward_marshal.hpp — alpha's batched-FORWARD capstone, PURE marshaling core.
//
// The deterministic front-half of the M>1 entry: turns the marshaled CSR view into
// (a) beta's BatchSchedule (per-request spans + req_of_token) and (b) the ForwardGraphKey
// that keys the encoded-CB cache for this fire's shape. No Metal, no decoder, no io[]
// fill — the FEED (delta's step_batch) and ALLOCATE (delta's build_bound_decode io[]→11)
// halves consume this. Reduces to {R=1,N=1,page_bucket=0,pure-decode} at M=1, so the
// sealed single-stream key is produced unchanged.
//
// Split (locked #mac): delta ALLOCATEs io[]→11 (sized by DecodeGeometry caps), alpha
// FILLs the CSR buffers per fire from this BatchSchedule (req_of_token straight into
// IoSlot::ReqOfToken; spans drive per-request KV/slot binds), beta BINDs at the published
// ordinals in decode_dispatch_mb.
//
// Throughput path = PURE-DECODE M>1 (N==R): req_of_token is the identity [0..R), RsSlotIds[R]
// binds directly to gdn_core_slotted (per-token == per-request), is_pure_decode=true. The
// rs_slot_ids[req_of_token[t]] per-token expansion is the mixed/prefill (N>R) follow-on.

#include <algorithm>
#include <cstdint>

#include "batch_schedule.hpp"
#include "decode_abi.hpp"

namespace pie::metal {

// Coarsen the batch's max per-request page count into a bucket so the CB cache doesn't
// thrash on every +1 page (matches CUDA forward_graph bucketing). 0 pages → bucket 0.
inline std::uint32_t page_bucket_of(const BatchSchedule& s) {
    std::uint32_t max_pages = 0;
    for (const auto& sp : s.spans) max_pages = std::max(max_pages, sp.num_pages);
    return (max_pages + PAGE_BUCKET_GRAN - 1) / PAGE_BUCKET_GRAN;
}

// Derive the CB-cache key from a built schedule. Pure; M=1 → {1,1,0,true}.
inline ForwardGraphKey key_of(const BatchSchedule& s) {
    ForwardGraphKey k;
    k.n_requests    = static_cast<std::uint32_t>(s.R);
    k.n_tokens      = static_cast<std::uint32_t>(s.N);
    k.page_bucket   = page_bucket_of(s);
    k.is_pure_decode = s.is_pure_decode;
    return k;
}

}  // namespace pie::metal
