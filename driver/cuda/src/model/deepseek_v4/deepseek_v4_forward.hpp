#pragma once

#include <cstdint>

#include "ops/attention_workspace.hpp"
#include "ops/attention_flashinfer.hpp"
#include "distributed.hpp"
#include "store/kv_cache.hpp"
#include "store/dsv4_compress_cache.hpp"
#include "model/deepseek_v4/deepseek_v4.hpp"
#include "ops/gemm.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver::model {

struct StageHooks;

struct DsV4ForwardCfg {
    int tp_size = 1;
    int tp_rank = 0;
    NcclComm* tp_comm = nullptr;
    bool emit_logits = true;
    // Mirrors LlamaLikeForwardCfg: when CUDA graphs are on, the FlashInfer
    // plan must be built in graph mode so the captured body re-reads the
    // plan buffers on replay instead of baking first-fire metadata in.
    bool decode_plan_cuda_graph = true;
};

struct DsV4Workspace {
    // Multi-stream HC residual
    DeviceTensor hc_residual;    // [N, hc_mult, H] BF16 — main state
    DeviceTensor y;              // [N, H] — single-stream scratch
    DeviceTensor norm_x;         // [N, H]
    DeviceTensor norm_y;         // [N, H]

    // HC mixing scratch
    DeviceTensor hc_mixes_f32;   // [N, hc_mult*H] F32 — RMSNorm'd residual for GEMM
    DeviceTensor hc_gemm_out;    // [N, mix_hc] F32 — GEMM output
    DeviceTensor hc_post_mix;    // [N, hc_mult] F32
    DeviceTensor hc_comb_mix;    // [N, hc_mult, hc_mult] F32
    DeviceTensor hc_head_gemm;   // [N, hc_mult] F32 — head GEMM output

    // Attention Q/KV path
    DeviceTensor q_a;            // [N, q_lora_rank]
    DeviceTensor q;              // [N, num_heads * head_dim]
    DeviceTensor kv;             // [N, head_dim]  (single KV head)
    DeviceTensor q_rope;         // [N, num_heads * qk_rope_head_dim]
    DeviceTensor k_rope;         // [N, 1 * qk_rope_head_dim]
    DeviceTensor attn_out;       // [N, num_heads * head_dim]

    // Output projection
    DeviceTensor wo_a_out;       // [N, o_groups * o_lora_rank]
    DeviceTensor wo_b_out;       // [N, H]

    // MoE
    DeviceTensor router_logits;  // [N, E]
    DeviceTensor topk_idx;       // [N, K] int32
    DeviceTensor topk_weights;   // [N, K] fp32
    DeviceTensor moe_out;        // [N, H]

    // Shared expert
    DeviceTensor shared_gate;    // [N, moe_I]
    DeviceTensor shared_up;      // [N, moe_I]
    DeviceTensor shared_act;     // [N, moe_I]
    DeviceTensor shared_out;     // [N, H]

    // Attention sink correction
    DeviceTensor attn_lse;       // [N, num_heads] fp32 — log-sum-exp from attention

    // ── Compressed attention (C4/C128 layers) ────────────────────────
    // Compressor projection scratch (reused per layer)
    DeviceTensor comp_kv_proj;    // [N, coff_max * head_dim] BF16 — wkv projection
    DeviceTensor comp_score_proj; // [N, coff_max * head_dim] BF16 — wgate projection
    // Compressed KV buffer (rewritten each layer; holds one layer's worth)
    DeviceTensor comp_kv;         // [max_comp_tokens, head_dim] BF16
    // Compressed attention output and LSE (per layer, reused)
    DeviceTensor comp_attn_out;   // [N, num_heads * head_dim] BF16
    DeviceTensor comp_attn_lse;   // [N, num_heads] F32
    // Per-entry compressor metadata, four packed [max_comp_tokens] I32 slices:
    // boundary_tok | boundary_pos | window_lo | rope_pos
    DeviceTensor comp_meta;       // [4 * max_comp_tokens] I32

    // FlashInfer plan for the per-layer sliding-window attention. Planned
    // once per forward (the geometry is identical on every layer) and
    // reused across layers.
    // Built by DsV4Model::prepare(), consumed by the (capturable) body.
    ops::PrefillPlanCachePtr swa_plan;
    bool swa_plan_valid = false;
    int swa_plan_tokens = -1;
    int swa_plan_requests = -1;

    // Staging for one FP8 -> BF16 dequantised `wo_a` group slice,
    // `[o_lora_rank, (num_heads * head_dim) / o_groups]`. It has its own
    // allocation because `wo_a` is REPLICATED across TP ranks while every
    // routed-expert buffer shrinks as `moe_intermediate_size / tp_size`:
    // borrowing `expert_gate_w` for it (as this used to) fit at tp<=4 and
    // overran it at tp=8, which surfaced only as an illegal memory access
    // several kernels later.
    DeviceTensor wo_a_dequant;

    // Routed expert scratch
    DeviceTensor expert_in;      // [N, H] bf16 — gathered input rows
    DeviceTensor expert_gate_w;  // [moe_I, H] bf16 — dequanted gate weight
    DeviceTensor expert_up_w;    // [moe_I, H] bf16 — dequanted up weight
    DeviceTensor expert_down_w;  // [H, moe_I] bf16 — dequanted down weight
    DeviceTensor expert_gate;    // [N, moe_I] bf16 — gate output
    DeviceTensor expert_up;      // [N, moe_I] bf16 — up output
    DeviceTensor expert_out;     // [N, H] bf16 — expert output
    DeviceTensor route_idx;      // [N*K] int32 — token indices for one expert
    DeviceTensor route_w;        // [N*K] fp32 — routing weights

    // Device-side aligned MoE scratch (vLLM/SGL-style expert blocking).
    DeviceTensor aligned_route_ids;   // [aligned_rows] int32
    DeviceTensor aligned_expert_ids;  // [max_blocks]   int32
    DeviceTensor aligned_expert_in;   // [aligned_rows, H]
    DeviceTensor aligned_gate_up;     // [aligned_rows, 2*moe_I]
    DeviceTensor aligned_act;         // [aligned_rows, moe_I]
    DeviceTensor aligned_out;         // [aligned_rows, H]
    DeviceTensor a_gu_ptrs, b_gu_ptrs, c_gu_ptrs;
    DeviceTensor a_dn_ptrs, b_dn_ptrs, c_dn_ptrs;
    int aligned_block_size = 0;
    int aligned_max_blocks = 0;

    // Logits
    DeviceTensor logits;         // [O, vocab]

    static DsV4Workspace allocate(
        const HfConfig& cfg,
        int max_tokens,
        int max_logit_rows,
        int tp_size);
};

std::size_t dsv4_workspace_bytes(
    const HfConfig& cfg,
    int max_tokens,
    int max_logit_rows,
    int tp_size);

struct DsV4PlanState {
    // FlashInfer plan for SWA layers
    void* flashinfer_plan = nullptr;
};

void dsv4_forward_paged(
    const DsV4Weights& w,
    const HfConfig& cfg,
    const DsV4ForwardCfg& fwd_cfg,
    DsV4Workspace& ws,
    KvCache& kv_cache,
    DsV4CompressCache& comp_cache,
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
    int total_tokens,
    int num_requests,
    bool is_pure_decode,
    const std::uint8_t* row_valid_d,
    const std::int32_t* logit_row_indices_d = nullptr,
    int num_logit_rows = 0,
    // The fire's stage hooks (`ForwardInputs::stage_hooks`). Null = no
    // program attached.
    const StageHooks* hooks = nullptr);

}  // namespace pie_cuda_driver::model
