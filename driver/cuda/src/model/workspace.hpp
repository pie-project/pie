#pragma once

// De-facto universal per-fire forward workspace (`Workspace`) and its
// byte-budget helper (`workspace_bytes`), shared by every llama-like
// forward path. Originally scoped to Qwen3; the standalone prefill/paged
// Qwen3 forward functions that once lived here were superseded by the
// wire-driven `llama_like` forward and removed.

#include <cstdint>

#include "device_buffer.hpp"
#include "model/loaded_model.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver::model {

constexpr int workspace_mtp_draft_row_base(int max_tokens) {
    return max_tokens;
}

constexpr int workspace_logits_rows(
    int max_tokens, int max_mtp_draft_rows) {
    return max_tokens +
        (max_mtp_draft_rows > 0 ? max_mtp_draft_rows : 0);
}

// Reusable scratch buffers, sized once for `max_tokens`. The forward pass
// only writes prefixes of these, so reusing across calls is safe as long as
// you don't exceed `max_tokens`.
struct Workspace {
    // Stage-2 MTP: extra rows reserved at the TAIL of `logits` (beyond the
    // `max_tokens` target rows) to hold the K native MTP draft-logit rows
    // an `Intrinsic::MtpLogits` [K,vocab] binding reads. `mtp_draft_row_base` is
    // the first reserved row; drafts live at [base, base+K) and never collide
    // with the target rows [0, max_tokens). A program may request at most
    // 32 drafts, while the aggregate batch reserve is one row per possible
    // output row so several MTP programs can coexist.
    static constexpr int kMtpDraftRowsPerProgram = 32;
    int mtp_draft_row_base = 0;
    int mtp_draft_row_capacity = 0;

    DeviceTensor y;          // [max_tokens, hidden]
    DeviceTensor norm_x;     // [max_tokens, hidden]
    DeviceTensor spec_hidden; // [max_tokens, hidden] saved verifier hidden rows
    DeviceTensor qkv_fused;  // [max_tokens, Hq + 2*Hk]   — only allocated when fused
                             // QKV path is in use; empty otherwise.
    DeviceTensor rope_table; // [max_tokens, head_dim] FP32; first half of
                             // each row is standard-RoPE cos, second is sin.
    DeviceTensor q;          // [max_tokens, h_q  * head_dim]   — packed
    DeviceTensor k;          // [max_tokens, h_kv * head_dim]   — packed
    DeviceTensor v;          // [max_tokens, h_kv * head_dim]   — packed
    DeviceTensor attn_out;   // [max_tokens, h_q  * head_dim]   — packed
    DeviceTensor norm_y;     // [max_tokens, hidden]
    DeviceTensor gate_up_fused; // [max_tokens, 2*I] — fused gate+up output, empty
                                // when unfused
    DeviceTensor mtp_concat;    // [max_tokens, 2*hidden] — Qwen3.6 MTP fc input
    DeviceTensor mtp_row0_save; // [1, vocab] preserves target row 0 while MTP drafts run
    DeviceTensor gate;       // [max_tokens, intermediate]
    DeviceTensor up;         // [max_tokens, intermediate]
    DeviceTensor logits;     // [max_tokens, vocab]
    DeviceTensor probs;      // [max_tokens, vocab] FP32 — softmax scratch for sampling

    // Padded variants for the attention kernel when `head_dim_kernel >
    // head_dim` (Phi-3 ships head_dim=96; flashinfer's TC kernel only
    // works at {64, 128, 256, 512}, so we round up to 128). Empty
    // (numel()==0) for every other model — the forward graph aliases
    // the packed buffers directly.
    DeviceTensor q_padded;        // [max_tokens, h_q  * head_dim_kernel]
    DeviceTensor k_padded;        // [max_tokens, h_kv * head_dim_kernel]
    DeviceTensor v_padded;        // [max_tokens, h_kv * head_dim_kernel]
    DeviceTensor attn_out_padded; // [max_tokens, h_q  * head_dim_kernel]

    static Workspace allocate(const HfConfig& cfg, int max_tokens);

    // Variant for architectures whose per-layer MLP `intermediate_size`
    // exceeds the base `cfg.intermediate_size` (Gemma-4's
    // `use_double_wide_mlp` doubles the width on shared layers).
    // Caller passes the worst-case value; ws.gate / ws.up / logits are
    // sized accordingly. Other shapes match the standard `allocate`.
    static Workspace allocate_with_max_intermediate(
        const HfConfig& cfg, int max_tokens, int max_intermediate,
        int max_output_rows = -1);

    // Variant for architectures whose per-layer attention dimensions
    // (Hq = num_q_heads * head_dim, Hk = num_kv_heads * head_dim) vary
    // across layers — Gemma-4's full-attention layers run at
    // head_dim_global=512 while sliding layers run at head_dim=256, so
    // a single ws.q sized at the sliding width overflows on full
    // layers. Caller passes the worst-case `Hq` and `Hk`.
    static Workspace allocate_full(
        const HfConfig& cfg, int max_tokens,
        int max_intermediate, int max_Hq, int max_Hk,
        int max_output_rows = -1,
        int max_mtp_draft_rows = 0);
};

// Byte budget for the per-fire Workspace tensors, parameterized by
// the HF config and the per-fire token/output shape. Used by the memory
// planner to size the persistent workspace arena.
std::size_t workspace_bytes(const HfConfig& cfg,
                                  int N,
                                  int output_rows,
                                  int max_intermediate,
                                  int max_Hq,
                                  int max_Hk,
                                  int max_mtp_draft_rows = 0);

}  // namespace pie_cuda_driver::model
