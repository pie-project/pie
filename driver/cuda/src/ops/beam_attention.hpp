// Beam [B,P] attention integration (SEAM 1+3 orchestration, charlie's G2). See
// beam_attention.cpp for the full contract.
#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "attention_workspace.hpp"
#include "kv_cache.hpp"
#include "ops/beam_csrs.hpp"

namespace pie_cuda_driver::ops {

// Device buffers + host CSRs for one beam-decode attention step. Pages/WSlot are
// PHYSICAL page ids (slot→physical resolved upstream). CSR *_d/_h and packed_mask
// scratch are caller-provided (staged from `beam_build_csrs`, see beam_csrs.hpp).
struct BeamAttentionInputs {
    KvCacheLayerView layer;
    const void* q;              // [B, num_q_heads, head_dim] bf16
    const void* k_curr;         // [B, num_kv_heads, head_dim] bf16
    const void* v_curr;
    void* o;                    // [B, num_q_heads, head_dim] bf16 out

    const std::uint32_t* w_page_phys;   // [B] physical page id to write
    const std::uint32_t* w_off;         // [B] offset-in-page to write
    const std::uint8_t*  kvm_dense;     // [B, P*PAGE] bytes (0/1)
    const std::uint32_t* klen_d;        // [B] physical span (device)
    std::uint8_t*        packed_mask;   // out scratch, pre-zeroed, mask_indptr[B] bytes

    const std::uint32_t* qo_indptr_d;
    const std::uint32_t* kv_page_indices_d;
    const std::uint32_t* kv_page_indptr_d;
    const std::uint32_t* kv_last_page_lens_d;
    const std::int32_t*  mask_indptr_d;
    const std::uint32_t* qo_indptr_h;
    const std::uint32_t* kv_page_indptr_h;

    int B;
    int P;
    int page_size;
    int num_q_heads;
    AttentionWorkspace& attn_ws;
    cudaStream_t stream;
};

// Orchestrate SEAM 3 (write) → SEAM 1 (pack) → SEAM 1(a) (custom-mask prefill).
void beam_attention_forward(const BeamAttentionInputs& in);

}  // namespace pie_cuda_driver::ops
