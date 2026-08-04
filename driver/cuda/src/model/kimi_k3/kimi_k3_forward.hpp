#pragma once

// Kimi-K3 forward: a hybrid KDA/MLA decoder whose residual stream is not a
// stream but a *stack*.
//
// ── Attention residuals ────────────────────────────────────────────────
//
// Every layer carries a running sum (`prefix_sum`) and a stack of saved block
// residuals. Per layer, with `bs = attn_res_block_size`:
//
//     prefix = y
//     if blocks: y_in = blend(prefix, blocks, attn_res_{norm,proj})
//     else:      y_in = prefix
//     if li % bs == 0: push prefix onto blocks; prefix := 0
//     prefix += attn(rmsnorm(y_in))
//     y_in    = blend(prefix, blocks, mlp_res_{norm,proj})
//     prefix += mlp(rmsnorm(y_in))
//     y       = prefix
//
// and after the last layer `y = blend(y, blocks, output_attn_res_{norm,proj})`
// before the final norm. Note which value is pushed: the *unblended* incoming
// residual, while the layer itself consumes the blended one.
//
// ── Attention ──────────────────────────────────────────────────────────
//
// A `linear_attention` layer runs KDA: three projections, a depthwise causal
// conv with silu, a per-key-channel gated delta recurrence over a persistent
// [H, V, K] state, a sigmoid-gated RMSNorm and an output projection. A
// `full_attention` layer runs MLA with **no rotation** -- the rope halves are
// projected and concatenated but never rotated -- plus a sigmoid output gate.
//
// ── MLP ────────────────────────────────────────────────────────────────
//
// Layer 0 is a dense SiTU MLP. The rest are latent MoE: the hidden is
// down-projected to `routed_expert_hidden_size`, routed through the experts
// there, RMSNormed, projected back, and the shared expert (which reads the
// *full-width* hidden, not the latent) is added on top.

#include <cstdint>

#include "distributed.hpp"
#include "ops/attention_workspace.hpp"
#include "ops/gemm.hpp"
#include "store/kv_cache.hpp"
#include "store/mla_cache.hpp"
#include "store/recurrent_state_cache.hpp"
#include "model/kimi/kimi_forward.hpp"  // KimiPlanState (shared MLA plan type)
#include "model/kimi_k3/kimi_k3.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver::model {

struct KimiK3ForwardCfg {
    int tp_size = 1;
    NcclComm* tp_comm = nullptr;
    bool emit_logits = true;
};

struct KimiK3Workspace {
    // ── Residual bookkeeping ───────────────────────────────────────
    DeviceTensor y;          // [N, H] the running `prefix_sum`
    DeviceTensor hidden;     // [N, H] blended layer input
    DeviceTensor blocks;     // [num_blocks, N, H] the AttnRes stack
    DeviceTensor norm_x;     // [N, H]
    DeviceTensor norm_y;     // [N, H]
    DeviceTensor attn_out;   // [N, H] (also the TP all-reduce staging buffer)
    DeviceTensor mlp_out;    // [N, H]
    int num_blocks = 0;

    // ── MLA ────────────────────────────────────────────────────────
    DeviceTensor q_a;             // [N, q_lora]
    DeviceTensor q_b;             // [N, heads*(nope+rope)]
    DeviceTensor q_nope;          // [N, heads*nope]
    DeviceTensor q_pe;            // [N, heads*rope]
    DeviceTensor kv_a_mqa;        // [N, kv_lora+rope]
    DeviceTensor kv_c;            // [N, kv_lora]
    DeviceTensor k_pe;            // [N, rope]
    DeviceTensor q_nope_latent;   // [N, heads*kv_lora]
    DeviceTensor attn_latent;     // [N, heads*kv_lora]
    DeviceTensor attn_v;          // [N, heads*v_dim]
    DeviceTensor mla_gate;        // [N, heads*v_dim]

    // ── KDA ────────────────────────────────────────────────────────
    DeviceTensor kda_q, kda_k, kda_v;   // [N, W] bf16, W = local_kda_heads*D
    // Convolution outputs. Separate from the inputs because a depthwise
    // causal conv reads a K-wide window per output position.
    DeviceTensor kda_qc, kda_kc, kda_vc;  // [N, W] bf16
    DeviceTensor kda_g_raw;             // [N, W] bf16
    DeviceTensor kda_f_a;               // [N, D] bf16
    DeviceTensor kda_beta_raw;          // [N, Hk] bf16
    DeviceTensor kda_gproj;             // [N, W] bf16
    DeviceTensor kda_gate;              // [N, W] fp32
    DeviceTensor kda_beta;              // [N, Hk] fp32
    DeviceTensor kda_qf, kda_kf, kda_vf;  // [N, W] fp32
    DeviceTensor kda_o;                 // [N, W] fp32

    // ── MLP / MoE ──────────────────────────────────────────────────
    DeviceTensor gate, up;        // [N, max(dense_I, shared_I)]
    DeviceTensor latent_in;       // [N, L]
    DeviceTensor latent_out;      // [N, L]
    DeviceTensor router_logits;   // [N, E]
    DeviceTensor topk_idx;        // [N, K] int32
    DeviceTensor topk_weights;    // [N, K] fp32
    DeviceTensor expert_out;      // [N*K, L]
    DeviceTensor shared_out;      // [N, H]
    DeviceTensor aligned_route_ids;
    DeviceTensor aligned_expert_ids;
    DeviceTensor aligned_expert_in;   // [aligned_rows, L]
    DeviceTensor aligned_gate_up;     // [aligned_rows, 2*routed_I]
    DeviceTensor aligned_act;         // [aligned_rows, routed_I]
    DeviceTensor aligned_out;         // [aligned_rows, L]
    // The MXFP4 decode GEMVs consume their activation as fp16 so their inner
    // loop is pure `__hfma2`; these stage it once per MoE layer.
    DeviceTensor mxfp4_act_fp16;        // [N, L] fp16
    DeviceTensor mxfp4_route_act_fp16;  // [N*K, routed_I] fp16
    DeviceTensor a_gu_ptrs, b_gu_ptrs, c_gu_ptrs;
    DeviceTensor a_dn_ptrs, b_dn_ptrs, c_dn_ptrs;
    int aligned_block_size = 0;
    int aligned_max_blocks = 0;

    DeviceTensor logits;          // [O, vocab]

    static KimiK3Workspace allocate(
        const HfConfig& cfg,
        int max_tokens,
        int max_logit_rows,
        int tp_size);
};

std::size_t kimi_k3_workspace_bytes(
    const HfConfig& cfg,
    int max_tokens,
    int max_logit_rows,
    int tp_size);

/// Number of AttnRes blocks a model of this depth opens.
int kimi_k3_num_attn_res_blocks(const HfConfig& cfg);

void kimi_k3_forward_paged(
    const KimiK3Weights& w,
    const HfConfig& cfg,
    const KimiK3ForwardCfg& fwd_cfg,
    KimiPlanState& mla_plan,
    KimiK3Workspace& ws,
    MlaCache& mla_cache,
    RecurrentStateCache& kda_cache,
    AttentionWorkspace& attn_ws,
    ops::CublasHandle& cublas,
    void* logits_out,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    const std::int32_t*  kda_slot_ids_h,
    const std::int32_t*  kda_slot_ids_d,
    const std::uint8_t*  kda_is_fresh_h,
    const std::uint8_t*  kda_is_fresh_d,
    int total_tokens,
    int num_requests,
    bool is_pure_decode,
    const std::uint8_t* row_valid_d = nullptr,
    const std::int32_t* logit_row_indices_d = nullptr,
    int num_logit_rows = 0);

}  // namespace pie_cuda_driver::model
