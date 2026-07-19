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

namespace pie_cuda_driver::kernels {

// Maintenance: reduce each page's live keys to its min/max envelope.
// `k_pages` is `[num_pages, page_size, num_kv_heads, head_dim]` bf16 (NHD, as
// `std::uint16_t`); `page_live_lens[p]` is the number of live tokens in page p
// (`<= page_size`). `env_min`/`env_max` are `[num_pages, num_kv_heads, head_dim]`
// f32. A page with 0 live tokens leaves `env_min=+inf`, `env_max=-inf`.
void launch_envelope_recompute_bf16(
    const std::uint16_t* k_pages,
    const std::int32_t* page_live_lens,
    float* env_min,
    float* env_max,
    int num_pages,
    int page_size,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream);

// Score: `score[kv_head, page]` (f32 `[num_kv_heads, P_MAX]`) per the Quest
// formula above. `q` is `[num_q_heads, head_dim]` f32 (this layer's projected
// query, M=1 decode); `env_min`/`env_max` are `[P_MAX, num_kv_heads, head_dim]`
// f32. Pages `page >= live_pages` → `-inf`. Requires
// `num_q_heads % num_kv_heads == 0` (GQA group).
void launch_envelope_dot_f32(
    const float* q,
    const float* env_min,
    const float* env_max,
    float* score,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int p_max,
    int live_pages,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
