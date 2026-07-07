// Beam [B,P] attention integration (SEAM 1+3 orchestration, charlie's G2).
//
// Given the beam epilogue's per-beam geometry (Pages resolved to PHYSICAL page
// ids, WSlot/WOff, klen, dense kvm, np), this drives the full beam-decode
// attention: (1) write each beam's new-token K/V to its explicit (WSlot,WOff)
// target (SEAM 3); (2) pack the dense kvm into FlashInfer's custom-mask bitmap
// (SEAM 1); (3) run the custom-mask prefill with 1 query per beam (SEAM 1(a) —
// decode has no per-cell mask; the fork-freeze mid-page hole needs it).
//
// The per-beam CSR construction (`beam_build_csrs`) is pure host logic keyed on
// `np[b]` (NOT Pages==0 — slot 0 is a valid page id; alpha's guardrail #1), so it
// is unit-tested against the fork-freeze golden geometry independently.

#include "ops/beam_attention.hpp"

#include "cuda_check.hpp"
#include "kernels/beam_mask_adapter.hpp"
#include "kernels/kv_paged.hpp"
#include "ops/attention_flashinfer.hpp"

namespace pie_cuda_driver::ops {

void beam_attention_forward(const BeamAttentionInputs& in)
{
    if (in.B <= 0) return;

    // 1. SEAM 3 — write each beam's new-token K/V to its explicit (WSlot,WOff).
    kernels::launch_write_kv_beam_bf16(
        in.layer, in.k_curr, in.v_curr, in.w_page_phys, in.w_off, in.B, in.stream);

    // 2. SEAM 1 — pack the dense kvm into FlashInfer's custom-mask bitmap.
    kernels::launch_beam_pack_kvm(
        in.kvm_dense, in.klen_d, in.mask_indptr_d, in.packed_mask,
        in.B, in.P * in.page_size, in.stream);

    // 3. SEAM 1(a) — custom-mask prefill, 1 query per beam.
    launch_attention_flashinfer_prefill_custom(
        in.q, in.layer, in.o,
        in.qo_indptr_d, in.kv_page_indices_d, in.kv_page_indptr_d,
        in.kv_last_page_lens_d, in.packed_mask, in.mask_indptr_d,
        in.qo_indptr_h, in.kv_page_indptr_h,
        /*total_tokens=*/in.B, /*num_requests=*/in.B, in.num_q_heads,
        in.attn_ws, in.stream);
}

}  // namespace pie_cuda_driver::ops
