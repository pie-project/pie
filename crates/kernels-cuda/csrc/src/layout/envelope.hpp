#pragma once

// Quest page-envelope kernels (overview §6.1). A program running
// Quest attention selects the top-`budget` KV pages per step by an importance
// score derived from each page's per-dimension key min/max ENVELOPE:
//
//   score[kv_head, page] = Σ_{qh ∈ GQA-group(kv_head)} Σ_d
//                              max(q[qh,d]·min[page,kv_head,d],
//                                  q[qh,d]·max[page,kv_head,d])
//
// i.e. the MAX possible q·k achievable by any key inside the page's envelope
// (an upper bound on the page's attention logit — Quest criticality). Two
// standalone kernels, not attention-kernel modifications:
//
//   1. `envelope_recompute` — maintenance: reduce each page's live keys to the
//      per-(page, kv_head, dim) min/max envelope. (The KV-append incremental
//      form is a follow-up; recompute is the reference/opt-in maintenance.)
//   2. `envelope_dot` — the [num_kv_heads, P_MAX] importance score above; pages
//      beyond the live size are validity-coded `-inf` (never selected).
//
// The golden reference is `pie_sampling_ir::eval::envelope_dot_reference`; this
// kernel is parity-checked bit-for-bit against it (test_envelope_dot).
//
// NOTE (thin adapter, reconcile with §6.1's `envelope_dot` intrinsic binding):
// the GQA-group aggregation (SUM here) + whether the program consumes the
// per-`kv_head` `[num_kv_heads, P_MAX]` core or a further `[P_MAX]` reduction are
// the ONLY provisional knobs; the per-dim `max(q·min, q·max)` core is fixed.

#include <cstdint>

#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels::layout {

// The envelopes are stored bf16, half of what f32 cost, and the narrowing is
// EXACT rather than a tolerated approximation: every envelope entry is the min
// or max of a set of bf16 keys, so it is already representable. The maintenance
// kernels still round in a DIRECTED way (min down, max up) so that a caller who
// one day merges a non-bf16 value cannot round the bound inwards and turn an
// upper bound into an estimate.
//
// Seed every page to the EMPTY envelope (`+inf`, `-inf`). Used once, when the
// envelopes are allocated alongside a fresh KV pool: no page holds a key yet,
// so the empty envelope is the exact answer, and every subsequent append
// refreshes the pages it touched.
void launch_envelope_seed_empty_bf16(
    std::uint16_t* env_min,
    std::uint16_t* env_max,
    int num_pages,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream);

// Maintenance: reduce each page's live keys to its min/max envelope.
// `k_pages` is `[num_pages, page_size, num_kv_heads, head_dim]` bf16 (NHD, as
// `std::uint16_t`); `page_live_lens[p]` is the number of live tokens in page p
// (`<= page_size`). `env_min`/`env_max` are `[num_pages, num_kv_heads, head_dim]`
// bf16 (as `std::uint16_t`). A page with 0 live tokens leaves `env_min=+inf`,
// `env_max=-inf`.
void launch_envelope_recompute_bf16(
    const std::uint16_t* k_pages,
    const std::int32_t* page_live_lens,
    std::uint16_t* env_min,
    std::uint16_t* env_max,
    int num_pages,
    int page_size,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream);

// Maintenance for the EXPLICIT-descriptor KV write, where the program supplies
// one (physical page, offset) per token instead of a page CSR. `k_curr` is this
// fire's keys `[num_tokens, num_kv_heads, head_dim]` bf16; `w_page[t]`/`w_off[t]`
// are the physical page and in-page offset token `t` was written to;
// `row_valid` (optional) masks padded lanes. A page entered at offset 0 is
// reset first (it was recycled), then each written key is MERGED into its
// page's envelope -- which equals a full recompute for append-only pages and
// only ever widens the bound for the mid-page rewrites the explicit path
// exists to serve. See the kernel comments.
void launch_envelope_merge_written_bf16(
    const std::uint16_t* k_curr,
    const std::uint32_t* w_page,
    const std::uint32_t* w_off,
    const std::uint8_t* row_valid,
    std::uint16_t* env_min,
    std::uint16_t* env_max,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream);

// formula above. `q` is `[num_q_heads, head_dim]` f32 (this layer's projected
// query, M=1 decode); `env_min`/`env_max` are `[P_MAX, num_kv_heads, head_dim]`
// bf16. The dot itself is f32: the envelopes widen exactly (see
// `envelope_device.cuh`), so only their storage is narrow.
// Pages `page >= live_pages` → `-inf`. Requires
// `num_q_heads % num_kv_heads == 0` (GQA group).

// Maintenance: refresh the envelopes of exactly the pages a fire appended to,
// deriving that set on-device from the same CSR arithmetic `write_kv_kernel`
// uses (a request's post-append length is `(pages-1)*page_size +
// last_page_len`, so its new tokens are the last `qo_len` of that span). The KV
// write path only ever holds device pointers, which is why the touched set is
// derived in-kernel rather than passed as an explicit host-built list.
// Rescanning the whole page list instead would cost a full KV read per layer,
// i.e. as much as attention itself.
//
// `max_touched` is the host's worst-case bound on the number of touched pages,
// `ceil(total_tokens/page_size) + num_requests`; blocks past the true count exit
// without writing. Pages are append-only, so recomputing a touched page in full
// gives the same answer an incremental merge would. Shares its per-(page,
// kv_head) reduction with `launch_envelope_recompute_bf16`, so the numerics are
// the ones `test_envelope_dot` parity-checks. NHD layout only.
void launch_envelope_update_appended_bf16(
    const std::uint16_t* k_pages,
    const std::uint32_t* qo_indptr,          // [num_requests + 1]
    const std::uint32_t* kv_page_indices,    // [kv_page_indptr[num_requests]]
    const std::uint32_t* kv_page_indptr,     // [num_requests + 1]
    const std::uint32_t* kv_last_page_lens,  // [num_requests]
    std::uint16_t* env_min,
    std::uint16_t* env_max,
    int num_requests,
    int max_touched,
    int page_size,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels::layout
