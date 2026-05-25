#pragma once

// Gemma-4 (E2B / E4B family — text-only stripped of the multimodal
// towers). Architectural surface beyond Gemma-3:
//
//   * Per-Layer Embeddings (PLE): a 256-dim auxiliary residual stream
//     fed by `embed_tokens_per_layer` (one row per (token, layer))
//     plus a shared `per_layer_model_projection` of the main embed.
//     A per-layer (gate, projection, post-norm) triple injects this
//     signal back into the main residual stream after each MLP block.
//   * KV-cache sharing: the last `num_kv_shared_layers` layers reuse
//     K/V from the most recent non-shared layer of the same attention
//     type (sliding vs full). Shared layers store no K/V, run only a
//     Q-projection, and read attention through the source layer's
//     paged cache.
//   * Per-layer-type head_dim: sliding layers use `head_dim` (256 on
//     E2B), full-attention layers use `global_head_dim` (512). The
//     KV cache is allocated per-layer via `KvCache::allocate_per_layer`.
//   * Per-layer intermediate (`use_double_wide_mlp`): shared layers
//     have `intermediate_size * 2` MLP width.
//   * `sm_scale = 1.0` — Q/K-norm absorbs the usual `1/sqrt(d)` factor.
//   * Plain RMSNorm (`w * x_hat`, *not* Gemma-2's `(1+w) * x_hat`).
//   * V-Norm: pure RMSNorm (no learnable gamma) on V before the KV
//     write. Implemented as an RMSNorm with gamma=1.
//   * Per-layer learnable scalar: each layer's output is multiplied
//     by a scalar `layer_scalar`.
//   * Final logit soft-cap (cap=30).
//   * Proportional RoPE on full-attention layers (only the lower
//     `partial_rotary_factor * head_dim` dims are rotated). The first
//     pass of this implementation rotates the full head_dim — that's
//     a documented approximation; we'll wire `partial_rotary_factor`
//     into the rope kernel as a follow-up.
//
// Out of scope here: the multimodal towers (`audio_config`,
// `vision_config`) and the `enable_moe_block` MoE variant of Gemma-4.

#include <cstdint>
#include <vector>

#include "device_buffer.hpp"
#include "distributed.hpp"
#include "model/loaded_model.hpp"
#include "kv_cache.hpp"
#include "model/llama_like.hpp"
#include "model/qwen3.hpp"
#include "model/qwen3_forward.hpp"
#include "ops/attention_flashinfer.hpp"
#include "ops/gemm.hpp"

namespace pie_cuda_driver::model {

struct Gemma4LayerWeights {
    // Four RMSNorms — same placement as Gemma-2/3.
    const DeviceTensor* attn_norm_pre  = nullptr;  // input_layernorm
    const DeviceTensor* attn_norm_post = nullptr;  // post_attention_layernorm
    const DeviceTensor* mlp_norm_pre   = nullptr;  // pre_feedforward_layernorm
    const DeviceTensor* mlp_norm_post  = nullptr;  // post_feedforward_layernorm

    // Q is always present; K/V/V-norm are nullptr on shared layers.
    const DeviceTensor* q_proj = nullptr;
    const DeviceTensor* k_proj = nullptr;
    const DeviceTensor* v_proj = nullptr;
    const DeviceTensor* o_proj = nullptr;
    const DeviceTensor* qkv_proj_fused = nullptr;  // [Hq + 2 * Hk, hidden]
    const DeviceTensor* q_norm = nullptr;
    const DeviceTensor* k_norm = nullptr;

    // MLP. `intermediate` may differ per layer (double-wide on shared
    // layers when `use_double_wide_mlp` is set).
    const DeviceTensor* gate_proj = nullptr;
    const DeviceTensor* up_proj   = nullptr;
    const DeviceTensor* down_proj = nullptr;
    const DeviceTensor* gate_up_proj_fused = nullptr;  // [2 * intermediate, hidden]

    // ── Sparse-MoE block (Gemma-4 26B-A4B) ────────────────────────────
    // Runs in parallel with the dense MLP. All pointers nullptr on the
    // dense-only Gemma-4 family (E2B / E4B / 31B). The router is a
    // stand-alone module: RMSNorm-no-scale → channel-scale → 1/sqrt(H)
    // → linear → softmax → top-k → renorm → per-expert-scale.
    const DeviceTensor* router_proj            = nullptr;  // [E, H]
    const DeviceTensor* router_scale           = nullptr;  // [H]
    const DeviceTensor* router_per_expert_scale = nullptr; // [E]
    const DeviceTensor* moe_gate_up_proj       = nullptr;  // [E, 2*Im, H]
    const DeviceTensor* moe_down_proj          = nullptr;  // [E, H, Im]
    // Three extra layernorms specific to the MoE-on-top variant.
    const DeviceTensor* mlp_norm_post_dense    = nullptr;  // post_feedforward_layernorm_1
    const DeviceTensor* moe_norm_pre           = nullptr;  // pre_feedforward_layernorm_2
    const DeviceTensor* moe_norm_post          = nullptr;  // post_feedforward_layernorm_2

    // PLE per-layer triple.
    const DeviceTensor* ple_input_gate = nullptr;  // [hidden_per_layer_input, hidden]
    const DeviceTensor* ple_projection = nullptr;  // [hidden, hidden_per_layer_input]
    const DeviceTensor* ple_norm       = nullptr;  // [hidden]

    // Per-layer learnable scalar. 1-element bf16 tensor.
    const DeviceTensor* layer_scalar  = nullptr;
    float layer_scalar_value = 1.f;

    // Per-layer dimensions (filled in by `bind_gemma4` from layer_types).
    int head_dim     = 0;
    int intermediate = 0;
    int kv_source    = 0;   // == layer index when not shared
    int num_kv_heads = 0;   // 26B-A4B uses num_global_kv on full layers
    bool is_full     = false;
    bool is_shared   = false;
    // 26B-A4B "k_eq_v" mode: full-attention layers omit `v_proj.weight`
    // and derive V from the raw k_proj output (before k_norm / RoPE),
    // then v-norm. Sliding layers always have a separate v_proj.
    bool use_k_as_v  = false;
};

struct Gemma4Weights {
    const DeviceTensor* embed       = nullptr;       // model.language_model.embed_tokens.weight
    const DeviceTensor* embed_per_layer = nullptr;   // [vocab, num_layers * ple_dim]
    const DeviceTensor* ple_model_proj = nullptr;    // [num_layers * ple_dim, hidden]
    const DeviceTensor* ple_model_norm = nullptr;    // [ple_dim]
    const DeviceTensor* final_norm  = nullptr;       // model.language_model.norm
    const DeviceTensor* lm_head     = nullptr;       // tied to embed unless lm_head.weight present
    std::vector<Gemma4LayerWeights> layers;

    // Owned per-layer `router.scale` baked together with `1/sqrt(H)`,
    // so the router pipeline collapses to a single rmsnorm+weight call.
    // Empty on dense Gemma-4 ckpts; one tensor per layer when MoE is on.
    std::vector<DeviceTensor> owned_router_combined_scales;

    // Dense Gemma4 ships gate/up as separate tensors. Decode/spec batches are
    // small enough that launching two narrow GEMMs per layer is expensive, so
    // dense variants can materialize a packed [gate; up] tensor once at bind
    // time and issue a single wide GEMM in the hot path.
    std::vector<DeviceTensor> owned_gate_up_fused;

    // Dense non-shared Gemma4 layers can likewise materialize [q; k; v]
    // projection weights. This avoids two extra small-M GEMM launches on the
    // layers that actually write K/V.
    std::vector<DeviceTensor> owned_qkv_fused;

    // Cached per-layer arrays for the forward / KV-cache allocator.
    std::vector<int> per_layer_head_dim;
    std::vector<int> per_layer_intermediate;
    std::vector<int> per_layer_num_kv_heads;
    std::vector<int> kv_source_layer;
    std::vector<int> per_layer_window_left;
    std::vector<float> per_layer_rope_theta;
    std::vector<float> per_layer_partial_rotary_factor;
    std::vector<int> full_layer_indices;  // for debug / introspection
};

// MoE workspace for Gemma-4 26B-A4B's parallel routed-expert block.
// Inert (all buffers length 0) on dense Gemma-4 ckpts.
struct Gemma4MoeMlpWorkspace {
    // Dense Gemma-4 PLE inputs. Despite the historical struct name, these
    // are used by E2B/E4B too; keeping them here avoids a second Gemma4
    // workspace object threaded through the executor.
    DeviceBuffer<std::uint16_t> ple_token;         // token scratch, then [L, N, H_ple]
    DeviceBuffer<std::uint16_t> ple_proj;          // [N, L * H_ple]

    // Router intermediate buffers. `router_logits` holds the full E-way
    // softmax distribution; topk extracts the top-K weights/indices.
    DeviceBuffer<std::uint16_t> router_x;          // [N, H] post-norm input to router proj
    DeviceBuffer<std::uint16_t> router_logits;     // [N, E] bf16 (softmax output)
    DeviceBuffer<std::int32_t>  topk_idx;          // [N, K]
    DeviceBuffer<float>         topk_weights;      // [N, K]

    // MoE-branch input (= pre_feedforward_layernorm_2(residual)).
    DeviceBuffer<std::uint16_t> moe_input;         // [N, H]
    // Per-expert worst-case scratch.
    DeviceBuffer<std::uint16_t> expert_in;         // [N*K, H]
    DeviceBuffer<std::uint16_t> expert_gate_up;    // [N*K, 2*Im]
    DeviceBuffer<std::uint16_t> expert_act;        // [N*K, Im]
    DeviceBuffer<std::uint16_t> expert_out;        // [N*K, H]
    DeviceBuffer<std::int32_t>  expert_idx;        // [N*K]
    DeviceBuffer<float>         expert_w;          // [N*K]
    // Final accumulator before the post_feedforward_layernorm_2.
    DeviceBuffer<std::uint16_t> moe_out;           // [N, H]

    // Short causal multi-token verification (Gemma4 MTP) can be run as
    // one FlashInfer decode row per token. These buffers hold the expanded
    // per-row KV page table and avoid per-fire cudaMalloc churn.
    DeviceBuffer<std::uint32_t> row_decode_kv_page_indices;
    DeviceBuffer<std::uint32_t> row_decode_kv_page_indptr;
    DeviceBuffer<std::uint32_t> row_decode_kv_last_page_lens;
    std::vector<std::uint32_t> h_row_decode_kv_page_indices;
    std::vector<std::uint32_t> h_row_decode_kv_page_indptr;
    std::vector<std::uint32_t> h_row_decode_kv_last_page_lens;
    ops::DecodePlanCachePtr decode_plan_sliding;
    ops::DecodePlanCachePtr decode_plan_full;
    ops::DecodePlanCachePtr row_decode_plan_sliding;
    ops::DecodePlanCachePtr row_decode_plan_full;
    bool decode_plans_prepared = false;
    bool row_decode_prepared = false;
    int row_decode_prepared_tokens = 0;
    int row_decode_prepared_requests = 0;

    // Decode fast-path (N=1) batched-GEMM pointer arrays.
    DeviceBuffer<const std::uint16_t*> a_gu_ptrs;
    DeviceBuffer<const std::uint16_t*> b_gu_ptrs;
    DeviceBuffer<std::uint16_t*>       c_gu_ptrs;
    DeviceBuffer<const std::uint16_t*> a_dn_ptrs;
    DeviceBuffer<const std::uint16_t*> b_dn_ptrs;
    DeviceBuffer<std::uint16_t*>       c_dn_ptrs;
    DeviceBuffer<float>                batch_weights;

    static Gemma4MoeMlpWorkspace allocate(
        int max_tokens, int hidden, int num_experts, int top_k,
        int moe_intermediate);

    void allocate_row_decode(int max_tokens);
    void allocate_ple(int max_tokens, int per_layer_total);
};

struct Gemma4ForwardCfg {
    // Final logit soft-cap (defaults to Gemma-4-E2B's 30).
    float final_logit_softcap = 30.f;

    // Force the prefill kernel for batches whose GQA group size isn't
    // in flashinfer's decode dispatch table — same role as in
    // `LlamaLikeForwardCfg`.
    bool force_prefill_path = false;

    // TP state. tp_size > 1 activates per-layer sharded dims (each layer
    // shrinks Hq/Hk/I by tp_size) and inserts all-reduces after o_proj
    // and down_proj. Per-layer head_dim is unaffected — only the head
    // *count* divides.
    int tp_size = 1;
    NcclComm* tp_comm = nullptr;
};

// Bind helper: validates the Gemma-4 schema, populates per-layer
// dimensions + KV-source mapping from `HfConfig::layer_types` and
// `num_kv_shared_layers`. Throws on missing tensors.
Gemma4Weights bind_gemma4(const LoadedModel& engine);

void set_gemma4_logits_argmax_only(bool enabled);
void set_gemma4_fused_argmax_output(std::int32_t* ptr);
bool gemma4_fused_argmax_done();

void prepare_gemma4_decode_plans(
    const Gemma4Weights& w,
    const HfConfig& cfg,
    const Gemma4ForwardCfg& fwd_cfg,
    Gemma4MoeMlpWorkspace& moe_ws,
    KvCache& cache,
    AttentionWorkspace& attn_ws,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indices_h,
    const std::uint32_t* kv_page_indptr_h,
    const std::uint32_t* kv_last_page_lens_h,
    int total_tokens,
    int num_requests,
    bool is_pure_decode);

std::uint32_t gemma4_decode_graph_layout(
    const Gemma4MoeMlpWorkspace& moe_ws);

void gemma4_forward_paged(
    const Gemma4Weights& w,
    const HfConfig& cfg,
    const Gemma4ForwardCfg& fwd_cfg,
    Qwen3Workspace& ws,
    Gemma4MoeMlpWorkspace& moe_ws,
    KvCache& cache,
    AttentionWorkspace& attn_ws,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indices_h,
    const std::uint32_t* kv_page_indptr_h,
    const std::uint32_t* kv_last_page_lens_h,
    int total_tokens,
    int num_requests,
    bool is_pure_decode,
    const std::uint8_t* custom_mask_d = nullptr,
    const std::int32_t* custom_mask_indptr_d = nullptr,
    const std::int32_t* logit_row_indices_d = nullptr,
    int num_logit_rows = 0);

}  // namespace pie_cuda_driver::model
