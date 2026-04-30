#include "forward.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include "msgpack.hpp"
#include "sampler.hpp"

namespace pie_ggml_driver {

namespace {

// ggml_flash_attn_ext requires the mask's row count to be a multiple of
// GGML_KQ_MASK_PAD (= 64). Pad accordingly when building per-request and
// packed-decode masks.
constexpr std::int32_t MASK_PAD = 64;
// Per-call ggml graph node budget. Sized to comfortably fit MoE + spec
// decode batches (every layer's MoE op count is ~10-20 nodes).
constexpr std::size_t GRAPH_MAX_NODES = 1ull << 19;

// (legacy mul_with_cast removed; norm_scale is the single entry point now)

// RMSNorm scale step. Applies `x * w` for L4MA, or `x * (1 + w)` for the
// Gemma family (their RMSNorm is centered at 1; weights are stored as
// `actual_weight - 1`). Cast handles BF16 / F16 weights against F32
// activations.
ggml_tensor* norm_scale(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w,
                        bool plus_one) {
    if (w->type != x->type) {
        w = ggml_cast(ctx, w, x->type);
    }
    if (plus_one) {
        // ggml_scale_bias(t, scale, bias) → t * scale + bias.
        // We want (w + 1.0), so scale=1, bias=1.
        w = ggml_scale_bias(ctx, w, 1.0f, 1.0f);
    }
    return ggml_mul(ctx, x, w);
}

ggml_tensor* add_with_cast(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w) {
    if (w->type != x->type) {
        w = ggml_cast(ctx, w, x->type);
    }
    return ggml_add(ctx, x, w);
}

// Build the F16 KQ mask required by `ggml_flash_attn_ext` for one request.
// Layout: `[n_kv, n_tokens_pad]`. 0.0 where token i can attend to position
// j (j <= positions[i]); -INF otherwise (including padding rows).
//
// `runs_for_token(i)` (when non-null) is an optional per-token BRLE
// override (M6 custom attention masks). When provided, that row of the
// mask is built from the BRLE runs instead of the default causal pattern.
// The BRLE alternates false/true starting with false, like logit masks.
// Causal beyond positions[i] is still enforced (BRLE can't unmask the
// future).
struct PerTokenMaskRuns {
    const std::uint32_t* runs;
    std::size_t          n_runs;
};

void build_attn_mask_f16(std::vector<std::uint16_t>& dst,
                         std::int32_t n_kv,
                         std::int32_t n_tokens,
                         std::int32_t n_tokens_pad,
                         const std::int32_t* positions,
                         const PerTokenMaskRuns* per_token_runs /* nullable */,
                         std::int32_t sliding_window /* 0 = causal */ = 0) {
    dst.assign(static_cast<std::size_t>(n_kv) * n_tokens_pad,
               ggml_fp32_to_fp16(-INFINITY));
    const auto zero = ggml_fp32_to_fp16(0.0f);
    for (std::int32_t i = 0; i < n_tokens; ++i) {
        const std::int32_t p_i = positions[i];
        std::uint16_t* row = dst.data() + static_cast<std::size_t>(i) * n_kv;
        const std::int32_t hi = std::min(n_kv - 1, p_i);
        // Sliding window cuts off the past below `p_i - W + 1`.
        const std::int32_t lo =
            (sliding_window > 0) ? std::max(0, p_i - sliding_window + 1) : 0;

        if (!per_token_runs || per_token_runs[i].n_runs == 0) {
            for (std::int32_t j = lo; j <= hi; ++j) row[j] = zero;
            continue;
        }

        // Custom BRLE over [0, p_i]. Anything past p_i stays -INF. SWA
        // also clips below `lo`.
        const auto* runs   = per_token_runs[i].runs;
        const auto  n_runs = per_token_runs[i].n_runs;
        bool is_true = false;
        std::int32_t pos = 0;
        for (std::size_t k = 0; k < n_runs && pos <= hi; ++k) {
            const std::int32_t len = static_cast<std::int32_t>(runs[k]);
            const std::int32_t end = std::min(pos + len, hi + 1);
            if (is_true) {
                const std::int32_t a = std::max(pos, lo);
                for (std::int32_t j = a; j < end; ++j) row[j] = zero;
            }
            pos = end;
            is_true = !is_true;
        }
    }
}

// Backward-compat shim for the offline test plans: causal-only, no SWA.
void build_causal_mask_f16(std::vector<std::uint16_t>& dst,
                           std::int32_t n_kv,
                           std::int32_t n_tokens,
                           std::int32_t n_tokens_pad,
                           const std::int32_t* positions) {
    build_attn_mask_f16(dst, n_kv, n_tokens, n_tokens_pad, positions, nullptr, 0);
}

// Mixture-of-Experts FFN block.
//
// Inputs:
//   `cur`            : [hidden, n_total]                    (post-norm activation)
//   `gate_inp`       : [hidden, n_experts]                  (router weight)
//   `gate_exps`      : [hidden, ff,    n_experts]           (stacked SwiGLU gate)
//   `up_exps`        : [hidden, ff,    n_experts]           (stacked SwiGLU up)
//   `down_exps`      : [ff,     hidden, n_experts]          (stacked output proj)
//   `n_experts`      : total expert count
//   `n_used`         : top-k routing
//   `use_silu`       : true for SwiGLU (Mixtral / Qwen-MoE), false for GeGLU
//   `norm_topk`      : renormalize selected weights to sum to 1
//
// Returns: [hidden, n_total] — the per-token weighted sum over selected
// experts. Mirrors `src/llama-graph.cpp::llm_graph_context::build_moe_ffn`,
// pared down to the SwiGLU/GeGLU softmax-routing common case.
ggml_tensor* build_moe_ffn(ggml_context* ctx,
                           ggml_tensor* cur,
                           ggml_tensor* gate_inp,
                           ggml_tensor* gate_exps,
                           ggml_tensor* up_exps,
                           ggml_tensor* down_exps,
                           std::int32_t n_experts,
                           std::int32_t n_used,
                           bool use_silu,
                           bool norm_topk) {
    const std::int64_t n_total = cur->ne[1];

    // ---- 1. Router ------------------------------------------------------
    // logits[e, t] = sum_h gate_inp[h, e] * cur[h, t]   →  [n_experts, n_total]
    ggml_tensor* logits = ggml_mul_mat(ctx, gate_inp, cur);
    ggml_tensor* probs  = ggml_soft_max(ctx, logits);    // [n_experts, n_total]

    // ---- 2. Top-K experts per token -------------------------------------
    // ggml_top_k returns indices, shape [n_used, n_total] (I32).
    ggml_tensor* selected = ggml_top_k(ctx, probs, n_used);

    // ---- 3. Gather selected weights -------------------------------------
    // probs as [1, n_experts, n_total] so get_rows can pick rows along the
    // n_experts axis per-token.
    ggml_tensor* probs_3d  = ggml_reshape_3d(ctx, probs, 1, n_experts, n_total);
    ggml_tensor* w_gather  = ggml_get_rows(ctx, probs_3d, selected);
    // w_gather: [1, n_used, n_total]  →  [n_used, n_total]
    ggml_tensor* weights   = ggml_reshape_2d(ctx, w_gather, n_used, n_total);

    if (norm_topk) {
        // Sum along n_used (ne[0]) — `ggml_sum_rows` reduces ne[0].
        ggml_tensor* sum = ggml_sum_rows(ctx, weights);  // [1, n_total]
        weights = ggml_div(ctx, weights, sum);
    }

    // ---- 4. Per-token, per-selected expert: gate / up / down ------------
    // mul_mat_id requires `b` to be 3D `[hidden, 1, n_total]` so that the
    // result has shape `[ff, n_used, n_total]`.
    ggml_tensor* cur_3d = ggml_reshape_3d(ctx, cur, cur->ne[0], 1, n_total);

    ggml_tensor* gate_out = ggml_mul_mat_id(ctx, gate_exps, cur_3d, selected);
    ggml_tensor* up_out   = ggml_mul_mat_id(ctx, up_exps,   cur_3d, selected);

    ggml_tensor* act    = use_silu ? ggml_silu(ctx, gate_out)
                                   : ggml_gelu(ctx, gate_out);
    ggml_tensor* gated  = ggml_mul(ctx, act, up_out);  // [ff, n_used, n_total]

    ggml_tensor* expert_out =
        ggml_mul_mat_id(ctx, down_exps, gated, selected);
    // expert_out: [hidden, n_used, n_total]

    // ---- 5. Apply routing weights and sum across n_used -----------------
    // Reshape weights to [1, n_used, n_total] for elementwise broadcast.
    ggml_tensor* w_3d = ggml_reshape_3d(ctx, weights, 1, n_used, n_total);
    ggml_tensor* weighted = ggml_mul(ctx, expert_out, w_3d);

    // Sum across the n_used dim (ne[1]) — easiest via permute then sum_rows.
    // Permute (0, 1, 2, 3) → (0, 2, 1, 3) puts n_used into ne[2]; we want
    // it innermost for sum_rows. So permute (1, 0, 2, 3) puts n_used at
    // ne[0]. But weighted is 3D [hidden, n_used, n_total]; we want
    // result [hidden, n_total]. Approach: ggml_cont + reshape to
    // [hidden*n_total, n_used]? Cleaner: use a manual reduction by
    // selecting and adding each n_used slice.
    //
    // Simplest correct: permute to [n_used, hidden, n_total], sum_rows
    // → [1, hidden, n_total], reshape to [hidden, n_total].
    ggml_tensor* perm = ggml_permute(ctx, weighted, 1, 0, 2, 3);
    ggml_tensor* perm_cont = ggml_cont(ctx, perm);
    ggml_tensor* summed = ggml_sum_rows(ctx, perm_cont);
    // summed: [1, hidden, n_total]
    return ggml_reshape_2d(ctx, summed, cur->ne[0], n_total);
}

// Translate a logical position `p` in the request's sequence into the
// physical row index in the flat KV pool.
//   page = page_indices[indptr_off + p / page_size]
//   slot = p % page_size
//   physical = page * page_size + slot
inline std::int64_t physical_idx(const std::uint32_t* page_indices,
                                 std::int32_t indptr_off,
                                 std::int32_t page_size,
                                 std::int32_t pos) {
    const std::int32_t page_offset = pos / page_size;
    const std::int32_t slot        = pos % page_size;
    const std::int32_t page        = static_cast<std::int32_t>(
        page_indices[indptr_off + page_offset]);
    return static_cast<std::int64_t>(page) * page_size + slot;
}


// =============================================================================
// Graph build (multi-request, paged KV)
// =============================================================================
//
// Single graph builder covers the L4MA family (qwen2, qwen3, llama3-style)
// with per-arch optional features:
//   - has_qkv_bias: qwen2 adds biases to Q/K/V projections
//   - has_qk_norm:  qwen3 RMSNorms Q and K (per-head) before RoPE
//
// Future arches (gemma, mistral SWA, MoE) get further flags or branches.
// Per-arch feature switches. Empty / zero / 0.0 = "feature off" by default.
struct ArchSpec {
    bool   has_qkv_bias       = false;  // qwen2, phi3 (sometimes)
    bool   has_qk_norm        = false;  // qwen3, gemma3
    bool   has_pre_ffn_norm   = false;  // gemma2/3/4 (extra norm before FFN)
    bool   has_post_attn_norm = false;  // gemma2/3/4 (extra norm after attn)
    bool   has_post_ffn_norm  = false;  // gemma2/3/4 (extra norm after FFN)
    bool   scale_embed_by_sqrt_d = false;  // gemma family
    float  attn_softcap       = 0.0f;   // gemma2 (50.0 typical), 0 = none
    float  final_softcap      = 0.0f;   // gemma2 (30.0 typical), 0 = none
    // Per-layer attention pattern. Empty = all-global (causal). Non-empty
    // = vector of size n_layers; entries 'g' (global) or 's' (sliding).
    // For mistral all layers are sliding. For gemma3, every Nth is 'g'.
    std::string layer_pattern;
    std::int32_t sliding_window = 0;     // 0 = no SWA
    // Custom Q scaling (gemma2/3 use 1/sqrt(query_pre_attn_scalar) instead
    // of 1/sqrt(head_dim)). 0 = use default head_dim.
    float  query_pre_attn_scalar = 0.0f;
    // MLP activation. SiLU (SwiGLU) for L4MA family; GeLU (GeGLU) for Gemma.
    bool   ffn_use_gelu = false;
    // Gemma family: RMSNorm uses `(1 + weight)` instead of `weight`. The
    // weights are stored centered at 0 around 1, so we add 1 before
    // multiplying the normalized activation.
    bool   norm_weight_plus_one = false;

    // ── MoE ──
    // n_experts == 0 means dense MLP (the standard SwiGLU/GeGLU path).
    std::int32_t n_experts        = 0;
    std::int32_t n_experts_per_tok = 0;
    bool         moe_norm_topk    = true;   // renormalize selected weights
};

inline ArchSpec arch_spec_for(PieArch a, const Hparams& h) {
    ArchSpec s;
    switch (a) {
        case PieArch::Qwen3:
            s.has_qk_norm = true;
            break;
        case PieArch::Qwen2:
            s.has_qkv_bias = true;
            break;
        case PieArch::Llama3:
            // No quirks beyond optional NTK-by-parts (handled via
            // `weights.freq_factors`, not here).
            break;
        case PieArch::Mistral3:
            // Mistral / Ministral: SWA on every layer (when set).
            if (h.sliding_window) {
                s.sliding_window = *h.sliding_window;
                s.layer_pattern.assign(h.num_hidden_layers, 's');
            }
            break;
        case PieArch::Phi3:
            // Phi3 has fused QKV in tensor names — handled at load time.
            break;
        case PieArch::Gemma2:
            s.has_pre_ffn_norm   = true;
            s.has_post_attn_norm = true;
            s.has_post_ffn_norm  = true;
            s.ffn_use_gelu       = true;  // GeGLU
            s.norm_weight_plus_one = true;
            if (h.query_pre_attn_scalar) {
                s.query_pre_attn_scalar = *h.query_pre_attn_scalar;
            }
            s.scale_embed_by_sqrt_d = true;
            if (h.attn_logit_softcapping) s.attn_softcap = *h.attn_logit_softcapping;
            if (h.final_logit_softcapping) s.final_softcap = *h.final_logit_softcapping;
            // Gemma2 alternates global/sliding per layer (every other).
            if (h.sliding_window) {
                s.sliding_window = *h.sliding_window;
                s.layer_pattern.resize(h.num_hidden_layers);
                for (std::int32_t i = 0; i < h.num_hidden_layers; ++i) {
                    s.layer_pattern[i] = (i % 2 == 0) ? 's' : 'g';
                }
            }
            break;
        case PieArch::Gemma3:
            s.has_qk_norm        = true;
            s.has_pre_ffn_norm   = true;
            s.has_post_attn_norm = true;
            s.has_post_ffn_norm  = true;
            s.scale_embed_by_sqrt_d = true;
            s.ffn_use_gelu       = true;
            s.norm_weight_plus_one = true;
            if (h.query_pre_attn_scalar) {
                s.query_pre_attn_scalar = *h.query_pre_attn_scalar;
            }
            // Gemma3 iSWA: every Nth layer is global; others are sliding.
            // The HF config provides `sliding_window_pattern` (default 6).
            // We default to all-global if no SWA set, else N=6 pattern.
            if (h.sliding_window) {
                s.sliding_window = *h.sliding_window;
                s.layer_pattern.resize(h.num_hidden_layers);
                for (std::int32_t i = 0; i < h.num_hidden_layers; ++i) {
                    // Pattern: every 6th layer is 'g', rest are 's'.
                    s.layer_pattern[i] = ((i + 1) % 6 == 0) ? 'g' : 's';
                }
            }
            break;
        case PieArch::Mixtral:
        case PieArch::GptOss:
        case PieArch::Qwen3_5:
            // MoE archs. Carry over per-arch attention features:
            //   - GptOss has attention sinks (handled in flash_attn_ext call
            //     via the model's optional sinks tensor; not yet wired in
            //     the graph builder).
            //   - All three use top-k softmax routing with renormalization.
            s.n_experts         = h.num_experts;
            s.n_experts_per_tok = h.num_experts_per_tok;
            s.moe_norm_topk     = h.norm_topk_prob;
            break;
        case PieArch::Gemma4:
            // Gemma4 details land via per-layer hparams plumbed through the
            // graph builder — see build_gemma4_(). PLE (Per-Layer
            // Embeddings) is a structural difference not yet implemented.
            s.has_pre_ffn_norm   = true;
            s.has_post_attn_norm = true;
            s.has_post_ffn_norm  = true;
            s.scale_embed_by_sqrt_d = true;
            s.ffn_use_gelu       = true;
            s.norm_weight_plus_one = true;
            if (h.sliding_window) {
                s.sliding_window = *h.sliding_window;
            }
            break;
        default:
            break;
    }
    return s;
}

// -----------------------------------------------------------------------------
// Plan helpers
//
// These split `ForwardEngine::plan_` into composable phases:
//   1. extract_plan_arrays:   pull all 23+ typed views from the BPIQ wire blob
//   2. validate_plan_top_level: check the top-level invariants (batch shape)
//   3. resolve_active_adapter_id: enforce single-adapter-per-batch (v1)
//   4. plan_single_request:   build one ReqPlan + per-token positions/kv idxs
//   5. build_pure_decode_packing: M11 fast-path packing for the all-decode case
// -----------------------------------------------------------------------------

struct PlanArrays {
    std::span<const std::uint64_t> context_ids;
    std::span<const std::uint32_t> token_ids;
    std::span<const std::uint32_t> position_ids;
    std::span<const std::uint32_t> qo_indptr;
    std::span<const std::uint32_t> sampling_idx;
    std::span<const std::uint32_t> sampling_indptr;
    std::span<const std::uint32_t> request_num_samplers;
    std::span<const std::uint32_t> kv_page_indices;
    std::span<const std::uint32_t> kv_page_indptr;
    std::span<const std::uint32_t> kv_last_lens;
    std::span<const std::uint32_t> sampler_types;
    std::span<const float>         sampler_temps;
    std::span<const std::uint32_t> sampler_top_k;
    std::span<const float>         sampler_top_p;
    std::span<const float>         sampler_min_p;
    std::span<const std::uint32_t> sampler_seeds;
    std::span<const std::uint32_t> sampler_label_ids;
    std::span<const std::uint32_t> sampler_label_indptr;
    std::span<const std::uint32_t> logit_masks;
    std::span<const std::uint32_t> logit_mask_indptr;
    std::span<const std::uint32_t> flat_attn_masks;
    std::span<const std::uint32_t> attn_mask_indptr;
    std::span<const std::int64_t>  adapter_indices;
    std::span<const std::uint32_t> spec_token_ids;
    std::span<const std::uint32_t> spec_position_ids;
    std::span<const std::uint32_t> spec_indptr;

    std::int32_t n_request = 0;
    std::int32_t total_n_tokens = 0;
    bool         batch_has_drafts = false;
    bool         batch_has_attn_masks = false;
};

PlanArrays extract_plan_arrays(const schema::DecodedRequest& req) {
    PlanArrays a;
    a.context_ids          = req.as<std::uint64_t>(schema::A_CONTEXT_IDS);
    a.token_ids            = req.as<std::uint32_t>(schema::A_TOKEN_IDS);
    a.position_ids         = req.as<std::uint32_t>(schema::A_POSITION_IDS);
    a.qo_indptr            = req.as<std::uint32_t>(schema::A_QO_INDPTR);
    a.sampling_idx         = req.as<std::uint32_t>(schema::A_SAMPLING_INDICES);
    a.sampling_indptr      = req.as<std::uint32_t>(schema::A_SAMPLING_INDPTR);
    a.request_num_samplers = req.as<std::uint32_t>(schema::A_REQUEST_NUM_SAMPLERS);
    a.kv_page_indices      = req.as<std::uint32_t>(schema::A_KV_PAGE_INDICES);
    a.kv_page_indptr       = req.as<std::uint32_t>(schema::A_KV_PAGE_INDPTR);
    a.kv_last_lens         = req.as<std::uint32_t>(schema::A_KV_LAST_PAGE_LENS);
    a.sampler_types        = req.as<std::uint32_t>(schema::A_SAMPLER_TYPES);
    a.sampler_temps        = req.as<float>(schema::A_SAMPLER_TEMPERATURES);
    a.sampler_top_k        = req.as<std::uint32_t>(schema::A_SAMPLER_TOP_K);
    a.sampler_top_p        = req.as<float>(schema::A_SAMPLER_TOP_P);
    a.sampler_min_p        = req.as<float>(schema::A_SAMPLER_MIN_P);
    a.sampler_seeds        = req.as<std::uint32_t>(schema::A_SAMPLER_SEEDS);
    a.sampler_label_ids    = req.as<std::uint32_t>(schema::A_SAMPLER_LABEL_IDS);
    a.sampler_label_indptr = req.as<std::uint32_t>(schema::A_SAMPLER_LABEL_INDPTR);
    a.logit_masks          = req.as<std::uint32_t>(schema::A_LOGIT_MASKS);
    a.logit_mask_indptr    = req.as<std::uint32_t>(schema::A_LOGIT_MASK_INDPTR);
    a.flat_attn_masks      = req.as<std::uint32_t>(schema::A_FLATTENED_MASKS);
    a.attn_mask_indptr     = req.as<std::uint32_t>(schema::A_MASK_INDPTR);
    a.adapter_indices      = req.as<std::int64_t>(schema::A_ADAPTER_INDICES);
    a.spec_token_ids       = req.as<std::uint32_t>(schema::A_SPEC_TOKEN_IDS);
    a.spec_position_ids    = req.as<std::uint32_t>(schema::A_SPEC_POSITION_IDS);
    a.spec_indptr          = req.as<std::uint32_t>(schema::A_SPEC_INDPTR);

    a.n_request      = static_cast<std::int32_t>(a.context_ids.size());
    a.total_n_tokens = static_cast<std::int32_t>(a.token_ids.size());
    a.batch_has_drafts =
        a.spec_indptr.size() == static_cast<std::size_t>(a.n_request) + 1;
    // Custom (non-causal) masks are present iff the indptr fully covers
    // every token's row. Otherwise the M11 packed-decode fast path is safe.
    a.batch_has_attn_masks =
        !a.flat_attn_masks.empty() &&
        a.attn_mask_indptr.size() ==
            static_cast<std::size_t>(a.total_n_tokens) + 1;
    return a;
}

void validate_plan_top_level(const PlanArrays& a) {
    if (a.n_request == 0) {
        throw std::runtime_error("plan: no requests in batch");
    }
    if (a.token_ids.size() != a.position_ids.size()) {
        throw std::runtime_error("plan: token_ids/position_ids size mismatch");
    }
    const auto n_plus_1 = static_cast<std::size_t>(a.n_request) + 1;
    if (a.qo_indptr.size() != n_plus_1) {
        throw std::runtime_error("plan: qo_indptr length must be num_requests+1");
    }
    if (a.kv_page_indptr.size() != n_plus_1) {
        throw std::runtime_error("plan: kv_page_indptr length must be num_requests+1");
    }
    if (a.kv_last_lens.size() != static_cast<std::size_t>(a.n_request)) {
        throw std::runtime_error("plan: kv_last_page_lens length must equal num_requests");
    }
    if (a.request_num_samplers.size() != static_cast<std::size_t>(a.n_request)) {
        throw std::runtime_error("plan: request_num_samplers length mismatch");
    }
    if (a.sampling_indptr.size() != n_plus_1) {
        throw std::runtime_error("plan: sampling_indptr length must be num_requests+1");
    }
}

// Returns the active adapter id (-1 if no request set one). Throws if
// requests in the same batch ask for different adapters — v1 enforces
// a single LoRA per fire_batch.
std::int64_t resolve_active_adapter_id(const PlanArrays& a) {
    std::int64_t active = -1;
    bool any = false;
    for (auto x : a.adapter_indices) {
        if (x < 0) continue;
        if (!any) { active = x; any = true; }
        else if (x != active) {
            throw std::runtime_error(
                "plan: mixed adapters in one batch — v1 supports a single "
                "adapter per fire_batch (got " + std::to_string(active) +
                " and " + std::to_string(x) + ")");
        }
    }
    return active;
}

void plan_single_request(const PlanArrays& a,
                         std::int32_t r,
                         std::int32_t page_size,
                         std::int32_t total_pages,
                         const ArchSpec& spec,
                         ForwardEngine::BatchPlan& plan) {
    // M8 spec decode allows multi-slot per request. v1 expects ≥1 slot.
    if (a.request_num_samplers[r] < 1) {
        throw std::runtime_error(
            "plan: request " + std::to_string(r) + " has 0 sampler slots");
    }
    const std::int32_t qo_start = static_cast<std::int32_t>(a.qo_indptr[r]);
    const std::int32_t qo_end   = static_cast<std::int32_t>(a.qo_indptr[r + 1]);
    const std::int32_t n_tok    = qo_end - qo_start;
    if (n_tok <= 0) {
        throw std::runtime_error(
            "plan: request " + std::to_string(r) + " has zero tokens");
    }

    const std::int32_t pages_off   = static_cast<std::int32_t>(a.kv_page_indptr[r]);
    const std::int32_t num_pages_r =
        static_cast<std::int32_t>(a.kv_page_indptr[r + 1]) - pages_off;
    const std::int32_t last_len    = static_cast<std::int32_t>(a.kv_last_lens[r]);
    const std::int32_t seq_len     = num_pages_r > 0
        ? (num_pages_r - 1) * page_size + last_len
        : last_len;
    if (seq_len <= 0) {
        throw std::runtime_error(
            "plan: request " + std::to_string(r) + " has empty KV state");
    }

    // Sanity-check that page IDs are in-bounds for the configured pool.
    for (std::int32_t pi = 0; pi < num_pages_r; ++pi) {
        const auto p = a.kv_page_indices[pages_off + pi];
        if (p >= static_cast<std::uint32_t>(total_pages)) {
            throw std::runtime_error(
                "plan: page id " + std::to_string(p) +
                " out of bounds (total_pages=" +
                std::to_string(total_pages) + ")");
        }
    }

    // BPIQ provides 1 slot per request (the last-pending position that
    // predicts the first draft / next token). Spec decode grows this to
    // 1 + n_drafts.
    const std::int32_t s_start = static_cast<std::int32_t>(a.sampling_indptr[r]);
    const std::int32_t s_end   = static_cast<std::int32_t>(a.sampling_indptr[r + 1]);
    const std::int32_t n_bpiq_slots = s_end - s_start;
    if (n_bpiq_slots < 1) {
        throw std::runtime_error(
            "plan: request " + std::to_string(r) +
            " has 0 BPIQ sampling indices; expected ≥1");
    }
    const std::int32_t primary_idx =
        static_cast<std::int32_t>(a.sampling_idx[s_start]);
    if (primary_idx < qo_start || primary_idx >= qo_end) {
        throw std::runtime_error(
            "plan: request " + std::to_string(r) +
            " sampling_index " + std::to_string(primary_idx) +
            " out of [" + std::to_string(qo_start) + "," +
            std::to_string(qo_end) + ")");
    }

    // Per-token: tokens, positions, and write idxs.
    for (std::int32_t i = qo_start; i < qo_end; ++i) {
        const std::int32_t pos_i = static_cast<std::int32_t>(a.position_ids[i]);
        if (pos_i >= seq_len) {
            throw std::runtime_error(
                "plan: request " + std::to_string(r) +
                " token at position " + std::to_string(pos_i) +
                " exceeds seq_len " + std::to_string(seq_len));
        }
        plan.tokens_i32[i]    = static_cast<std::int32_t>(a.token_ids[i]);
        plan.positions_i32[i] = pos_i;
        plan.kv_idxs_i64[i] = physical_idx(a.kv_page_indices.data(),
                                           pages_off, page_size, pos_i);
    }

    ForwardEngine::ReqPlan rp;
    rp.qo_start     = qo_start;
    rp.n_tokens     = n_tok;
    rp.n_tokens_pad = ((n_tok + MASK_PAD - 1) / MASK_PAD) * MASK_PAD;
    rp.n_kv         = seq_len;

    // Sampling slots: primary first, then one per draft.
    rp.sampling_positions.push_back(primary_idx);
    plan.sampling_pos_i32.push_back(primary_idx);

    if (a.batch_has_drafts) {
        const std::int32_t d_start = static_cast<std::int32_t>(a.spec_indptr[r]);
        const std::int32_t d_end   = static_cast<std::int32_t>(a.spec_indptr[r + 1]);
        const std::int32_t n_drafts = d_end - d_start;
        if (n_drafts > 0) {
            rp.draft_tokens.assign(a.spec_token_ids.data() + d_start,
                                   a.spec_token_ids.data() + d_end);
            // Each draft sits at the unique batch index i where
            // positions_i32[i] == draft_pos.
            for (std::int32_t k = 0; k < n_drafts; ++k) {
                const std::int32_t draft_pos =
                    static_cast<std::int32_t>(a.spec_position_ids[d_start + k]);
                std::int32_t idx = -1;
                for (std::int32_t i = qo_start; i < qo_end; ++i) {
                    if (static_cast<std::int32_t>(a.position_ids[i]) == draft_pos) {
                        idx = i; break;
                    }
                }
                if (idx < 0) {
                    throw std::runtime_error(
                        "plan: spec draft at position " + std::to_string(draft_pos) +
                        " not found in request " + std::to_string(r) + "'s qo range");
                }
                rp.sampling_positions.push_back(idx);
                plan.sampling_pos_i32.push_back(idx);
            }
        }
    }

    rp.gather_idxs.resize(seq_len);
    for (std::int32_t k = 0; k < seq_len; ++k) {
        rp.gather_idxs[k] = static_cast<std::int32_t>(
            physical_idx(a.kv_page_indices.data(), pages_off, page_size, k));
    }

    // Custom per-token attention masks (M6) — optional.
    std::vector<PerTokenMaskRuns> per_token_runs;
    if (a.batch_has_attn_masks) {
        per_token_runs.resize(n_tok);
        for (std::int32_t i = 0; i < n_tok; ++i) {
            const std::int32_t global_i = qo_start + i;
            const std::int32_t lo = static_cast<std::int32_t>(a.attn_mask_indptr[global_i]);
            const std::int32_t hi = static_cast<std::int32_t>(a.attn_mask_indptr[global_i + 1]);
            per_token_runs[i].runs   = a.flat_attn_masks.data() + lo;
            per_token_runs[i].n_runs = static_cast<std::size_t>(hi - lo);
        }
    }
    build_attn_mask_f16(rp.mask_f16, seq_len, n_tok, rp.n_tokens_pad,
                        plan.positions_i32.data() + qo_start,
                        a.batch_has_attn_masks ? per_token_runs.data() : nullptr,
                        spec.sliding_window);

    // Sampler params (slot index = s_start in v1, where each request has 1 slot).
    if (s_start >= static_cast<std::int32_t>(a.sampler_types.size())) {
        throw std::runtime_error(
            "plan: sampler_types too short for request " + std::to_string(r));
    }
    rp.sampler.type = static_cast<SamplerType>(a.sampler_types[s_start]);
    auto opt_at = [&](auto span, auto fallback) {
        return s_start < static_cast<std::int32_t>(span.size())
            ? span[s_start] : fallback;
    };
    rp.sampler.temperature = opt_at(a.sampler_temps, 1.0f);
    rp.sampler.top_k       = opt_at(a.sampler_top_k, 0u);
    rp.sampler.top_p       = opt_at(a.sampler_top_p, 1.0f);
    rp.sampler.min_p       = opt_at(a.sampler_min_p, 0.0f);
    rp.sampler.seed        = opt_at(a.sampler_seeds, 0u);

    // Per-slot Logprob/Logprobs labels (sampler-slot-keyed indptr).
    if (a.sampler_label_indptr.size() > static_cast<std::size_t>(s_start) + 1) {
        const std::int32_t la = static_cast<std::int32_t>(a.sampler_label_indptr[s_start]);
        const std::int32_t lb = static_cast<std::int32_t>(a.sampler_label_indptr[s_start + 1]);
        if (lb > la) {
            rp.sampler.labels.assign(
                a.sampler_label_ids.data() + la,
                a.sampler_label_ids.data() + lb);
        }
    }

    // Per-request BRLE logit mask. Optional — empty = no constraint.
    if (a.logit_mask_indptr.size() == static_cast<std::size_t>(a.n_request) + 1) {
        const std::int32_t lm_start = static_cast<std::int32_t>(a.logit_mask_indptr[r]);
        const std::int32_t lm_end   = static_cast<std::int32_t>(a.logit_mask_indptr[r + 1]);
        if (lm_end > lm_start) {
            rp.logit_mask_runs.assign(
                a.logit_masks.data() + lm_start,
                a.logit_masks.data() + lm_end);
        }
    }

    plan.reqs.push_back(std::move(rp));
}

// M11 packed-decode fast path. Caller has already verified every request
// has n_tokens == 1 and there are no custom attention masks, so the whole
// batch can be expressed as a single ne33-broadcast attention call per
// layer. Builds packed gather idxs + a single mask tensor of shape
// [max_n_kv, 64, 1, n_request].
void build_pure_decode_packing(ForwardEngine::BatchPlan& plan,
                               std::int32_t n_request,
                               const ArchSpec& spec) {
    const std::int32_t M = plan.max_n_kv;
    const std::int32_t N = n_request;
    plan.packed_gather_idxs.assign(static_cast<std::size_t>(M) * N, 0);
    plan.packed_mask_f16.assign(
        static_cast<std::size_t>(M) * MASK_PAD * N,
        ggml_fp32_to_fp16(-INFINITY));
    const auto zero = ggml_fp32_to_fp16(0.0f);
    const std::int32_t W = spec.sliding_window;
    for (std::int32_t r = 0; r < N; ++r) {
        const auto& rp = plan.reqs[r];
        for (std::int32_t k = 0; k < rp.n_kv; ++k) {
            plan.packed_gather_idxs[
                static_cast<std::size_t>(r) * M + k] = rp.gather_idxs[k];
        }
        // For stream r, row b=0 (the single query at pos = n_kv_r-1)
        // attends [max(0, n_kv_r - W), n_kv_r) when SWA; else full range.
        std::uint16_t* row = plan.packed_mask_f16.data()
            + static_cast<std::size_t>(r) * M * MASK_PAD;
        const std::int32_t lo = (W > 0) ? std::max(0, rp.n_kv - W) : 0;
        for (std::int32_t k = lo; k < rp.n_kv; ++k) {
            row[k] = zero;
        }
    }
}

struct GraphInputs {
    ggml_tensor*              tok_input;   // I32 [total_n_tokens]
    ggml_tensor*              pos_input;   // I32 [total_n_tokens]
    ggml_tensor*              kv_idxs;     // I64 [total_n_tokens] (write idxs)
    ggml_tensor*              out_idx;     // I32 [n_request]
    // Slow path (per-request): one mask + gather tensor per request.
    std::vector<ggml_tensor*> masks;       // F16 [n_kv_r, n_tokens_pad_r]
    std::vector<ggml_tensor*> gather_idxs; // I32 [n_kv_r] gather idxs per req
    // Fast path (pure-decode, M11): packed gather + mask, single attn call.
    ggml_tensor*              packed_gather = nullptr; // I32 [n_req * max_n_kv]
    ggml_tensor*              packed_mask   = nullptr; // F16 [max_n_kv, 64, 1, n_req]
};

struct GraphResult {
    ggml_cgraph* gf;
    ggml_tensor* logits;      // F32 [vocab_size, n_request]
    GraphInputs  in;
};

GraphResult build_qwen3_graph(ggml_context* ctx,
                              const Model& model,
                              KvCachePaged& kv,
                              const ForwardEngine::BatchPlan& plan) {
    const auto& h = model.hparams();
    const auto& w = model.weights();
    const std::int32_t head_dim   = h.head_dim;
    const std::int32_t n_q_heads  = h.num_attention_heads;
    const std::int32_t n_kv_heads = h.num_key_value_heads;
    const std::int32_t n_embd_gqa = n_kv_heads * head_dim;
    const std::int32_t n_total    = plan.total_n_tokens;
    const std::int32_t n_req      = static_cast<std::int32_t>(plan.reqs.size());
    const ArchSpec spec = arch_spec_for(h.arch, h);

    // Graph node budget: each layer adds ~10 ops per request (mostly attention
    // + concat) plus ~10 shared ops (norm, projections, FFN). 512 requests
    // over 28 layers needs ~150k nodes; budget 4× that to leave headroom.
    const int graph_size = static_cast<int>(GRAPH_MAX_NODES);
    auto* gf = ggml_new_graph_custom(ctx, graph_size, /*grads=*/ false);

    GraphInputs in{};
    in.tok_input = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_total);
    ggml_set_name(in.tok_input, "inp_tokens");
    ggml_set_input(in.tok_input);

    in.pos_input = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_total);
    ggml_set_name(in.pos_input, "inp_pos");
    ggml_set_input(in.pos_input);

    in.kv_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_total);
    ggml_set_name(in.kv_idxs, "kv_write_idxs");
    ggml_set_input(in.kv_idxs);

    // out_idx may exceed n_req when M8 spec decode adds per-draft slots.
    const std::int32_t n_sample_slots =
        static_cast<std::int32_t>(plan.sampling_pos_i32.size());
    in.out_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_sample_slots);
    ggml_set_name(in.out_idx, "out_idx");
    ggml_set_input(in.out_idx);

    if (plan.pure_decode) {
        // M11 fast path: a single packed gather + mask covers all requests.
        in.packed_gather = ggml_new_tensor_1d(
            ctx, GGML_TYPE_I32, static_cast<std::int64_t>(plan.max_n_kv) * n_req);
        ggml_set_name(in.packed_gather, "kv_gather.packed");
        ggml_set_input(in.packed_gather);

        in.packed_mask = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F16, plan.max_n_kv,
            static_cast<std::int64_t>(MASK_PAD), 1, n_req);
        ggml_set_name(in.packed_mask, "kq_mask.packed");
        ggml_set_input(in.packed_mask);
    } else {
        in.masks.reserve(n_req);
        in.gather_idxs.reserve(n_req);
        for (std::int32_t r = 0; r < n_req; ++r) {
            const auto& R = plan.reqs[r];
            auto* m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, R.n_kv, R.n_tokens_pad, 1, 1);
            ggml_set_name(m, ("kq_mask." + std::to_string(r)).c_str());
            ggml_set_input(m);
            in.masks.push_back(m);

            auto* g = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, R.n_kv);
            ggml_set_name(g, ("kv_gather." + std::to_string(r)).c_str());
            ggml_set_input(g);
            in.gather_idxs.push_back(g);
        }
    }

    // ---- Embed ---------------------------------------------------------------
    auto* embd = ggml_get_rows(ctx, w.tok_embd, in.tok_input);
    auto* inpL = embd;

    // Gemma family: multiply the embedding by sqrt(hidden_size) before
    // entering the layers (matches HF transformers' Gemma forward).
    if (spec.scale_embed_by_sqrt_d) {
        const float embed_scale = std::sqrt(static_cast<float>(h.hidden_size));
        inpL = ggml_scale(ctx, inpL, embed_scale);
    }

    // Default Q scaling = 1/sqrt(head_dim). Gemma2/3 override with
    // 1/sqrt(query_pre_attn_scalar). 0 = use head_dim default.
    const float kq_scale =
        (spec.query_pre_attn_scalar > 0.0f)
            ? 1.0f / std::sqrt(spec.query_pre_attn_scalar)
            : 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (std::int32_t il = 0; il < h.num_hidden_layers; ++il) {
        const auto& L = w.layers[il];
        auto* inpSA = inpL;

        auto* cur = ggml_rms_norm(ctx, inpL, h.rms_norm_eps);
        cur = norm_scale(ctx, cur, L.attn_norm, spec.norm_weight_plus_one);

        auto* Q = ggml_mul_mat(ctx, L.q_proj, cur);
        auto* K = ggml_mul_mat(ctx, L.k_proj, cur);
        auto* V = ggml_mul_mat(ctx, L.v_proj, cur);

        // M9: optional LoRA delta for q/k/v/o projections.
        //   y_lora = scale * (B @ (A @ x))
        // where A: [hidden, rank], B: [rank, out_dim].
        if (plan.active_adapter
            && static_cast<std::size_t>(il) < plan.active_adapter->layers().size()) {
            const auto& AL = plan.active_adapter->layers()[il];
            const float adapter_scale = plan.active_adapter->scale();
            auto apply_lora = [&](ggml_tensor* y,
                                  ggml_tensor* a, ggml_tensor* b) {
                if (!a || !b) return y;
                auto* a_out = ggml_mul_mat(ctx, a, cur);     // [rank, n_total]
                auto* b_out = ggml_mul_mat(ctx, b, a_out);   // [out, n_total]
                if (adapter_scale != 1.0f) {
                    b_out = ggml_scale(ctx, b_out, adapter_scale);
                }
                return ggml_add(ctx, y, b_out);
            };
            Q = apply_lora(Q, AL.q_a, AL.q_b);
            K = apply_lora(K, AL.k_a, AL.k_b);
            V = apply_lora(V, AL.v_a, AL.v_b);
        }

        // Optional QKV bias (qwen2). 1D bias vector broadcasts along ne[1]
        // (the n_total token dim) — same as flashinfer / HF.
        if (spec.has_qkv_bias) {
            if (L.q_proj_b) Q = add_with_cast(ctx, Q, L.q_proj_b);
            if (L.k_proj_b) K = add_with_cast(ctx, K, L.k_proj_b);
            if (L.v_proj_b) V = add_with_cast(ctx, V, L.v_proj_b);
        }

        Q = ggml_reshape_3d(ctx, Q, head_dim, n_q_heads,  n_total);
        K = ggml_reshape_3d(ctx, K, head_dim, n_kv_heads, n_total);
        V = ggml_reshape_3d(ctx, V, head_dim, n_kv_heads, n_total);

        // Optional QK-norm (qwen3). Per-head RMSNorm with weight broadcast
        // over n_q_heads / n_kv_heads.
        if (spec.has_qk_norm) {
            Q = ggml_rms_norm(ctx, Q, h.rms_norm_eps);
            Q = norm_scale(ctx, Q, L.q_norm, spec.norm_weight_plus_one);
            K = ggml_rms_norm(ctx, K, h.rms_norm_eps);
            K = norm_scale(ctx, K, L.k_norm, spec.norm_weight_plus_one);
        }

        // freq_factors: precomputed per-dim scaling for LLaMA-3.1+ NTK
        // RoPE. nullptr → plain θ-only RoPE (qwen2/qwen3/llama3.0).
        ggml_tensor* c_rope = w.freq_factors;
        // Gemma3 uses a different RoPE base on sliding-window layers
        // (rope_local_base_freq, typically 10000) vs global (rope_theta,
        // typically 1000000). All other archs use rope_theta everywhere.
        const bool is_sliding_layer =
            !spec.layer_pattern.empty()
            && static_cast<std::size_t>(il) < spec.layer_pattern.size()
            && spec.layer_pattern[il] == 's';
        const float layer_rope_theta =
            (is_sliding_layer && h.rope_local_base_freq > 0.0f)
                ? h.rope_local_base_freq
                : h.rope_theta;
        Q = ggml_rope_ext(ctx, Q, in.pos_input, c_rope,
                          head_dim, GGML_ROPE_TYPE_NEOX, /*n_ctx_orig=*/ 0,
                          layer_rope_theta, /*freq_scale=*/ 1.0f,
                          /*ext_factor=*/ 0.0f, /*attn_factor=*/ 1.0f,
                          /*beta_fast=*/ 32.0f, /*beta_slow=*/ 1.0f);
        K = ggml_rope_ext(ctx, K, in.pos_input, c_rope,
                          head_dim, GGML_ROPE_TYPE_NEOX, /*n_ctx_orig=*/ 0,
                          layer_rope_theta, /*freq_scale=*/ 1.0f,
                          /*ext_factor=*/ 0.0f, /*attn_factor=*/ 1.0f,
                          /*beta_fast=*/ 32.0f, /*beta_slow=*/ 1.0f);

        // ---- KV pool write (set_rows scatters by physical row index) -------
        auto* k_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, K), n_embd_gqa, n_total);
        auto* v_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, V), n_embd_gqa, n_total);

        auto* k_cached = ggml_set_rows(ctx, kv.k(il), k_2d, in.kv_idxs);
        auto* v_cached = ggml_set_rows(ctx, kv.v(il), v_2d, in.kv_idxs);

        // ---- Attention -----------------------------------------------------
        ggml_tensor* attn_2d = nullptr;

        if (plan.pure_decode) {
            // Packed: single flash_attn_ext per layer with ne3 = n_request.
            // Q is [head_dim, n_q_heads, n_total] with n_total == n_req.
            // Reshape (no data move; same memory layout) to
            // [head_dim, 1, n_q_heads, n_request] for ne3 broadcast.
            auto* Q_4d = ggml_reshape_4d(ctx, Q, head_dim, 1, n_q_heads, n_req);

            // Gather all requests' K/V in one call. Result is
            // [n_embd_gqa, max_n_kv * n_req] F32.
            auto* K_gather = ggml_get_rows(ctx, k_cached, in.packed_gather);
            auto* V_gather = ggml_get_rows(ctx, v_cached, in.packed_gather);

            // Reshape to [head_dim, n_kv_heads, max_n_kv, n_req], then
            // permute to [head_dim, max_n_kv, n_kv_heads, n_req] for
            // flash_attn_ext.
            auto* K_4d = ggml_reshape_4d(ctx, K_gather,
                                         head_dim, n_kv_heads,
                                         plan.max_n_kv, n_req);
            auto* V_4d = ggml_reshape_4d(ctx, V_gather,
                                         head_dim, n_kv_heads,
                                         plan.max_n_kv, n_req);
            auto* K_perm = ggml_permute(ctx, K_4d, 0, 2, 1, 3);
            auto* V_perm = ggml_permute(ctx, V_4d, 0, 2, 1, 3);

            auto* attn = ggml_flash_attn_ext(ctx, Q_4d, K_perm, V_perm,
                                             in.packed_mask, kq_scale,
                                             /*max_bias=*/ 0.0f,
                                             /*logit_softcap=*/ spec.attn_softcap);
            ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
            // attn shape per ggml.h: [head_dim, n_q_heads, n_batch=1, n_req]
            attn_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, attn),
                                      head_dim * n_q_heads, n_req);
        } else {
            // Slow path: one flash_attn_ext per request, then concat.
            std::vector<ggml_tensor*> attn_out_per_req;
            attn_out_per_req.reserve(n_req);

            const std::size_t Q_stride_ne2 =
                static_cast<std::size_t>(head_dim) * n_q_heads *
                ggml_type_size(Q->type);

            for (std::int32_t r = 0; r < n_req; ++r) {
                const auto& R = plan.reqs[r];

                auto* Q_r = ggml_view_3d(ctx, Q,
                                         head_dim, n_q_heads, R.n_tokens,
                                         /*nb1=*/ Q->nb[1],
                                         /*nb2=*/ Q->nb[2],
                                         /*offset=*/ static_cast<std::size_t>(R.qo_start) * Q_stride_ne2);
                auto* Q_r_perm = ggml_permute(ctx, Q_r, 0, 2, 1, 3);

                auto* K_gather = ggml_get_rows(ctx, k_cached, in.gather_idxs[r]);
                auto* V_gather = ggml_get_rows(ctx, v_cached, in.gather_idxs[r]);

                auto* K_r = ggml_reshape_3d(ctx, K_gather, head_dim, n_kv_heads, R.n_kv);
                auto* V_r = ggml_reshape_3d(ctx, V_gather, head_dim, n_kv_heads, R.n_kv);
                auto* K_r_perm = ggml_permute(ctx, K_r, 0, 2, 1, 3);
                auto* V_r_perm = ggml_permute(ctx, V_r, 0, 2, 1, 3);

                auto* attn = ggml_flash_attn_ext(ctx, Q_r_perm, K_r_perm, V_r_perm,
                                                 in.masks[r], kq_scale,
                                                 /*max_bias=*/ 0.0f,
                                                 /*logit_softcap=*/ spec.attn_softcap);
                ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
                attn_out_per_req.push_back(ggml_cont(ctx, attn));
            }

            ggml_tensor* attn_concat = attn_out_per_req[0];
            for (std::int32_t r = 1; r < n_req; ++r) {
                attn_concat = ggml_concat(ctx, attn_concat, attn_out_per_req[r], /*dim=*/ 2);
            }
            attn_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, attn_concat),
                                      head_dim * n_q_heads, n_total);
        }

        auto* attn_out = ggml_mul_mat(ctx, L.o_proj, attn_2d);

        // M9 LoRA delta on o_proj.
        if (plan.active_adapter
            && static_cast<std::size_t>(il) < plan.active_adapter->layers().size()) {
            const auto& AL = plan.active_adapter->layers()[il];
            if (AL.o_a && AL.o_b) {
                const float s = plan.active_adapter->scale();
                auto* a_out = ggml_mul_mat(ctx, AL.o_a, attn_2d);
                auto* b_out = ggml_mul_mat(ctx, AL.o_b, a_out);
                if (s != 1.0f) b_out = ggml_scale(ctx, b_out, s);
                attn_out = ggml_add(ctx, attn_out, b_out);
            }
        }

        // Gemma family: extra norm after the attention block, before
        // the residual add.
        if (spec.has_post_attn_norm && L.post_attn_norm) {
            attn_out = ggml_rms_norm(ctx, attn_out, h.rms_norm_eps);
            attn_out = norm_scale(ctx, attn_out, L.post_attn_norm, spec.norm_weight_plus_one);
        }

        auto* ffn_in = ggml_add(ctx, attn_out, inpSA);

        // FFN — pre-FFN norm. For llama-style this is `post_attention_layernorm`;
        // for gemma it's `pre_feedforward_layernorm`. Both stored in L.ffn_norm.
        cur = ggml_rms_norm(ctx, ffn_in, h.rms_norm_eps);
        cur = norm_scale(ctx, cur, L.ffn_norm, spec.norm_weight_plus_one);

        ggml_tensor* ffn_out;
        if (spec.n_experts > 0) {
            // MoE dispatch (Mixtral / Qwen-MoE / GPT-OSS / DeepSeek-style).
            ffn_out = build_moe_ffn(ctx, cur,
                                    L.moe_router,
                                    L.moe_gate_exps,
                                    L.moe_up_exps,
                                    L.moe_down_exps,
                                    spec.n_experts,
                                    spec.n_experts_per_tok,
                                    /*use_silu=*/ !spec.ffn_use_gelu,
                                    /*norm_topk=*/ spec.moe_norm_topk);
        } else {
            // Dense SwiGLU / GeGLU.
            auto* gate = ggml_mul_mat(ctx, L.gate_proj, cur);
            auto* up   = ggml_mul_mat(ctx, L.up_proj,   cur);
            gate = spec.ffn_use_gelu ? ggml_gelu(ctx, gate)
                                     : ggml_silu(ctx, gate);
            auto* gated = ggml_mul(ctx, gate, up);
            ffn_out = ggml_mul_mat(ctx, L.down_proj, gated);
        }

        // Gemma family: extra norm after the FFN block, before the
        // second residual add.
        if (spec.has_post_ffn_norm && L.post_ffn_norm) {
            ffn_out = ggml_rms_norm(ctx, ffn_out, h.rms_norm_eps);
            ffn_out = norm_scale(ctx, ffn_out, L.post_ffn_norm, spec.norm_weight_plus_one);
        }

        inpL = ggml_add(ctx, ffn_out, ffn_in);
    }

    auto* cur = ggml_rms_norm(ctx, inpL, h.rms_norm_eps);
    cur = norm_scale(ctx, cur, w.output_norm, spec.norm_weight_plus_one);

    auto* sampled = ggml_get_rows(ctx, cur, in.out_idx);

    ggml_tensor* lm_head_w = h.tie_word_embeddings ? w.tok_embd : w.output_head;
    auto* logits = ggml_mul_mat(ctx, lm_head_w, sampled);

    // Gemma2: final logit softcap (50.0 / 30.0). y = c * tanh(x / c).
    if (spec.final_softcap > 0.0f) {
        logits = ggml_scale(ctx, logits, 1.0f / spec.final_softcap);
        logits = ggml_tanh(ctx, logits);
        logits = ggml_scale(ctx, logits, spec.final_softcap);
    }

    ggml_set_name(logits, "logits");
    ggml_set_output(logits);

    ggml_build_forward_expand(gf, logits);

    GraphResult res{};
    res.gf = gf;
    res.logits = logits;
    res.in = std::move(in);
    return res;
}

// =============================================================================
// BPIS response writers (flat + msgpack)
// =============================================================================
constexpr std::uint32_t RESP_MAGIC        = 0x42504953;  // 'BPIS'
constexpr std::uint32_t RESP_MODE_FLAT    = 0;
constexpr std::uint32_t RESP_MODE_MSGPACK = 1;

// True if any per-request output requires the msgpack response path:
// special-sampler payloads (Distribution/RawLogits/Logprob/Logprobs/
// Entropy) or variable-length token lists from M8 speculative decoding
// (anything other than length-1 `tokens` triggers msgpack since the
// flat schema requires a uniform per-request count to be encoded too).
inline bool needs_msgpack_mode(const std::vector<SamplerOutput>& outs) {
    for (const auto& o : outs) {
        for (const auto& s : o.special_slots) {
            if (s.has_dist || !s.raw_logits.empty() || !s.logprobs.empty()
                || s.has_entropy) {
                return true;
            }
        }
    }
    return false;
}

// Shared 16-byte BPIS response header. Both flat and msgpack responses
// start with this; only the `mode` field and the meaning of `total_tokens`
// differ. Returns the header size for chaining.
constexpr std::size_t BPIS_HEADER_SIZE = 16;

inline void write_bpis_header(std::span<std::uint8_t> dst,
                              std::uint32_t mode,
                              std::uint32_t n_req,
                              std::uint32_t total_tokens) {
    if (dst.size() < BPIS_HEADER_SIZE) {
        throw std::runtime_error("response: dst too small for BPIS header");
    }
    auto write_u32 = [](std::uint8_t* p, std::uint32_t v) {
        std::memcpy(p, &v, 4);
    };
    write_u32(dst.data() + 0,  RESP_MAGIC);
    write_u32(dst.data() + 4,  mode);
    write_u32(dst.data() + 8,  n_req);
    write_u32(dst.data() + 12, total_tokens);
}

// Emit a `BPIS` msgpack-mode response with one ForwardPassResponse per
// request. Field shapes mirror `runtime/src/inference/request.rs` and
// Pie's Python `write_response` (rmp_serde + serde derive accepts maps
// with field-name keys).
std::size_t write_msgpack_response(std::span<std::uint8_t> dst,
                                   const std::vector<SamplerOutput>& outs) {
    write_bpis_header(dst, RESP_MODE_MSGPACK,
                      static_cast<std::uint32_t>(outs.size()),
                      /*total_tokens=*/ 0);  // unused in msgpack mode

    MsgpackWriter w(dst.subspan(BPIS_HEADER_SIZE));
    // {"results": [ ... ]}
    w.map_header(1);
    w.str("results");
    w.array_header(outs.size());
    for (const auto& o : outs) {
        // Aggregate the per-slot special-sampler payloads for this
        // request into the flat ForwardPassResponse field shape.
        std::size_t n_dists = 0, n_raw = 0, n_logprobs = 0, n_entropies = 0;
        for (const auto& s : o.special_slots) {
            if (s.has_dist)               ++n_dists;
            if (!s.raw_logits.empty())    ++n_raw;
            if (!s.logprobs.empty())      ++n_logprobs;
            if (s.has_entropy)            ++n_entropies;
        }

        w.map_header(7);

        w.str("tokens");
        w.array_u32(std::span<const std::uint32_t>(o.tokens));

        w.str("dists");
        w.array_header(n_dists);
        for (const auto& s : o.special_slots) {
            if (!s.has_dist) continue;
            // Each entry is a 2-tuple (Vec<u32>, Vec<f32>) — array of size 2.
            w.array_header(2);
            w.array_u32(std::span<const std::uint32_t>(s.dist_ids));
            w.array_f32(std::span<const float>(s.dist_vals));
        }

        w.str("logits");
        w.array_header(n_raw);
        for (const auto& s : o.special_slots) {
            if (s.raw_logits.empty()) continue;
            w.bin(s.raw_logits.data(), s.raw_logits.size());
        }

        w.str("logprobs");
        w.array_header(n_logprobs);
        for (const auto& s : o.special_slots) {
            if (s.logprobs.empty()) continue;
            w.array_f32(std::span<const float>(s.logprobs));
        }

        w.str("entropies");
        w.array_header(n_entropies);
        for (const auto& s : o.special_slots) {
            if (!s.has_entropy) continue;
            w.f32(s.entropy);
        }

        // spec_tokens / spec_positions — empty (drafter for next iter
        // is a separate concern; the verifier emits accepted tokens
        // through `tokens` directly).
        w.str("spec_tokens");
        w.array_header(0);
        w.str("spec_positions");
        w.array_header(0);
    }
    return BPIS_HEADER_SIZE + w.size();
}

std::size_t write_flat_response(std::span<std::uint8_t> dst,
                                std::span<const std::uint32_t> tokens_per_req,
                                std::span<const std::uint32_t> all_tokens) {
    const std::size_t n_req        = tokens_per_req.size();
    const std::size_t total_tokens = all_tokens.size();
    const std::size_t need = BPIS_HEADER_SIZE + n_req * 4 + total_tokens * 4;

    if (dst.size() < need) {
        throw std::runtime_error(
            "response: dst too small (have " + std::to_string(dst.size()) +
            ", need " + std::to_string(need) + ")");
    }

    write_bpis_header(dst, RESP_MODE_FLAT,
                      static_cast<std::uint32_t>(n_req),
                      static_cast<std::uint32_t>(total_tokens));
    std::memcpy(dst.data() + BPIS_HEADER_SIZE,
                tokens_per_req.data(), n_req * 4);
    std::memcpy(dst.data() + BPIS_HEADER_SIZE + n_req * 4,
                all_tokens.data(), total_tokens * 4);
    return need;
}

// -----------------------------------------------------------------------------
// Compute helpers
// -----------------------------------------------------------------------------

// Stage all per-batch host arrays into the graph's input tensors. Picks
// between the slow (per-request) path and the M11 packed-decode fast path
// based on `plan.pure_decode`.
void upload_graph_inputs(const GraphResult& g,
                         const ForwardEngine::BatchPlan& plan) {
    ggml_backend_tensor_set(g.in.tok_input, plan.tokens_i32.data(), 0,
                            plan.tokens_i32.size() * sizeof(std::int32_t));
    ggml_backend_tensor_set(g.in.pos_input, plan.positions_i32.data(), 0,
                            plan.positions_i32.size() * sizeof(std::int32_t));
    ggml_backend_tensor_set(g.in.kv_idxs, plan.kv_idxs_i64.data(), 0,
                            plan.kv_idxs_i64.size() * sizeof(std::int64_t));
    ggml_backend_tensor_set(g.in.out_idx, plan.sampling_pos_i32.data(), 0,
                            plan.sampling_pos_i32.size() * sizeof(std::int32_t));
    if (plan.pure_decode) {
        ggml_backend_tensor_set(g.in.packed_gather,
                                plan.packed_gather_idxs.data(), 0,
                                plan.packed_gather_idxs.size() * sizeof(std::int32_t));
        ggml_backend_tensor_set(g.in.packed_mask,
                                plan.packed_mask_f16.data(), 0,
                                plan.packed_mask_f16.size() * sizeof(std::uint16_t));
    } else {
        for (std::size_t r = 0; r < plan.reqs.size(); ++r) {
            const auto& mask = plan.reqs[r].mask_f16;
            const auto& gat  = plan.reqs[r].gather_idxs;
            ggml_backend_tensor_set(g.in.masks[r], mask.data(), 0,
                                    mask.size() * sizeof(std::uint16_t));
            ggml_backend_tensor_set(g.in.gather_idxs[r], gat.data(), 0,
                                    gat.size() * sizeof(std::int32_t));
        }
    }
}

// Returns true if any slot produced a special-sampler payload
// (Distribution / RawLogits / Logprob / Logprobs / Entropy).
inline bool any_slot_special(const std::vector<SlotOutput>& slots) {
    for (const auto& s : slots) {
        if (s.has_dist || !s.raw_logits.empty()
            || !s.logprobs.empty() || s.has_entropy) {
            return true;
        }
    }
    return false;
}

// Sample every slot for one request (after BRLE logit-mask application).
std::vector<SlotOutput> sample_request_slots(const ForwardEngine::ReqPlan& rp,
                                             float* slots_logits_base,
                                             std::int32_t n_slots,
                                             std::int32_t vocab_size) {
    std::vector<SlotOutput> out(n_slots);
    for (std::int32_t s = 0; s < n_slots; ++s) {
        float* row = slots_logits_base + static_cast<std::size_t>(s) * vocab_size;
        if (!rp.logit_mask_runs.empty()) {
            apply_brle_logit_mask(row, vocab_size,
                                  rp.logit_mask_runs.data(),
                                  rp.logit_mask_runs.size());
        }
        sample_slot(row, vocab_size, rp.sampler, out[s]);
    }
    return out;
}

// Resolve per-request sampled slots into a final SamplerOutput. Three
// modes:
//   - special: hand through per-slot special payloads
//   - spec decode: walk drafts vs predictions, accept matching prefix +
//     1 bonus token
//   - plain: emit the single sampled token
void resolve_request_output(const ForwardEngine::ReqPlan& rp,
                            std::vector<SlotOutput>&& slot_out,
                            SamplerOutput& dst) {
    if (any_slot_special(slot_out)) {
        dst.special_slots = std::move(slot_out);
        return;
    }
    const std::int32_t n_drafts =
        static_cast<std::int32_t>(rp.draft_tokens.size());
    if (n_drafts == 0) {
        dst.tokens.push_back(slot_out[0].token);
        return;
    }
    // Spec verifier walk. Slot k predicts the token that should follow
    // draft k (slot 0 predicts the first draft itself). Accept the
    // matching prefix + 1 bonus.
    dst.tokens.reserve(n_drafts + 1);
    bool all_match = true;
    for (std::int32_t k = 0; k < n_drafts; ++k) {
        if (slot_out[k].token == rp.draft_tokens[k]) {
            dst.tokens.push_back(rp.draft_tokens[k]);
        } else {
            dst.tokens.push_back(slot_out[k].token);  // bonus replacing rejected
            all_match = false;
            break;
        }
    }
    if (all_match) {
        // Every draft accepted — append the bonus from the last slot.
        dst.tokens.push_back(slot_out[n_drafts].token);
    }
}

// Top-level batch sampler: walk the per-request slot output starting at
// `all_logits` (laid out flat as [n_slots, vocab_size]).
std::vector<SamplerOutput> sample_batch(const ForwardEngine::BatchPlan& plan,
                                        float* all_logits,
                                        std::int32_t vocab_size) {
    const std::int32_t n_req = static_cast<std::int32_t>(plan.reqs.size());
    std::vector<SamplerOutput> sampled(n_req);
    std::int32_t slot_off = 0;
    for (std::int32_t r = 0; r < n_req; ++r) {
        const auto& rp = plan.reqs[r];
        const std::int32_t n_slots =
            static_cast<std::int32_t>(rp.sampling_positions.size());
        float* base = all_logits + static_cast<std::size_t>(slot_off) * vocab_size;
        auto slot_out = sample_request_slots(rp, base, n_slots, vocab_size);
        resolve_request_output(rp, std::move(slot_out), sampled[r]);
        slot_off += n_slots;
    }
    return sampled;
}

}  // namespace

// =============================================================================
// ForwardEngine
// =============================================================================

ForwardEngine::ForwardEngine(Model& model,
                             std::int32_t total_pages,
                             std::int32_t page_size)
    : model_(model),
      kv_(model.backend(),
          model.hparams().num_hidden_layers,
          model.hparams().num_key_value_heads,
          model.hparams().head_dim,
          total_pages,
          page_size,
          GGML_TYPE_F16) {
    galloc_ = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(model.backend()));
    if (!galloc_) {
        throw std::runtime_error("forward: ggml_gallocr_new failed");
    }
}

ForwardEngine::~ForwardEngine() {
    if (galloc_) ggml_gallocr_free(galloc_);
}

// -----------------------------------------------------------------------------
// Plan: BPIQ → BatchPlan (real page-table)
// -----------------------------------------------------------------------------

ForwardEngine::BatchPlan ForwardEngine::plan_(const schema::DecodedRequest& req) {
    const auto& hpar = model_.hparams();
    const ArchSpec spec = arch_spec_for(hpar.arch, hpar);

    const PlanArrays arrays = extract_plan_arrays(req);
    validate_plan_top_level(arrays);

    BatchPlan plan;
    plan.total_n_tokens = arrays.total_n_tokens;
    plan.tokens_i32.resize(plan.total_n_tokens);
    plan.positions_i32.resize(plan.total_n_tokens);
    plan.kv_idxs_i64.resize(plan.total_n_tokens);
    // sampling_pos_i32 is FLAT across all requests' slots; reserve a
    // generous upper bound (handles spec decode's 1 + n_drafts case).
    plan.sampling_pos_i32.reserve(arrays.n_request * 4);
    plan.reqs.reserve(arrays.n_request);

    // M9 LoRA: at most one adapter per fire_batch in v1.
    const std::int64_t active_adapter_id = resolve_active_adapter_id(arrays);

    const std::int32_t page_size   = kv_.page_size();
    const std::int32_t total_pages = kv_.total_pages();
    for (std::int32_t r = 0; r < arrays.n_request; ++r) {
        plan_single_request(arrays, r, page_size, total_pages, spec, plan);
    }

    // M11 packed-decode fast path: all-decode (n_tokens == 1) batches with
    // no custom masks fuse into one attn call per layer.
    plan.pure_decode = !plan.reqs.empty() && !arrays.batch_has_attn_masks;
    plan.max_n_kv = 0;
    for (const auto& rp : plan.reqs) {
        plan.max_n_kv = std::max(plan.max_n_kv, rp.n_kv);
        if (rp.n_tokens != 1) plan.pure_decode = false;
    }
    if (plan.pure_decode) {
        build_pure_decode_packing(plan, arrays.n_request, spec);
    }

    // Resolve the active adapter via the pool. If lookup fails, the
    // adapter wasn't registered yet — fall through to base-model behavior.
    if (active_adapter_id >= 0 && adapters_) {
        plan.active_adapter = adapters_->get(
            static_cast<std::uint64_t>(active_adapter_id));
        if (!plan.active_adapter) {
            std::cerr << "[forward] adapter id " << active_adapter_id
                      << " not in pool — running without adapter\n";
        }
    }
    return plan;
}


// -----------------------------------------------------------------------------
// Test harness plan: simulate Pie's page allocator with contiguous pages
// starting at `page_offset`.
// -----------------------------------------------------------------------------

ForwardEngine::BatchPlan ForwardEngine::plan_test_simple_(
        std::span<const std::uint32_t> token_ids,
        std::span<const std::uint32_t> position_ids,
        std::int32_t sampling_pos,
        std::int32_t page_offset) {
    if (token_ids.size() != position_ids.size() || token_ids.empty()) {
        throw std::runtime_error("plan_test: token/position size mismatch or empty");
    }
    const std::int32_t n_tok = static_cast<std::int32_t>(token_ids.size());
    if (sampling_pos < 0 || sampling_pos >= n_tok) {
        throw std::runtime_error("plan_test: sampling_pos out of range");
    }

    std::int32_t max_pos = 0;
    for (auto p : position_ids) {
        max_pos = std::max(max_pos, static_cast<std::int32_t>(p));
    }
    const std::int32_t seq_len   = max_pos + 1;
    const std::int32_t page_size = kv_.page_size();
    const std::int32_t pages_n   = (seq_len + page_size - 1) / page_size;
    if (page_offset + pages_n > kv_.total_pages()) {
        throw std::runtime_error(
            "plan_test: page allocation exceeds total_pages");
    }

    BatchPlan plan;
    plan.total_n_tokens = n_tok;
    plan.tokens_i32.resize(n_tok);
    plan.positions_i32.resize(n_tok);
    plan.kv_idxs_i64.resize(n_tok);
    // Filled below alongside ReqPlan::sampling_positions.

    auto pos_to_phys = [&](std::int32_t p) -> std::int64_t {
        const std::int32_t page = page_offset + p / page_size;
        return static_cast<std::int64_t>(page) * page_size + (p % page_size);
    };

    for (std::int32_t i = 0; i < n_tok; ++i) {
        const std::int32_t p = static_cast<std::int32_t>(position_ids[i]);
        plan.tokens_i32[i]    = static_cast<std::int32_t>(token_ids[i]);
        plan.positions_i32[i] = p;
        plan.kv_idxs_i64[i]   = pos_to_phys(p);
    }

    ReqPlan rp;
    rp.qo_start     = 0;
    rp.n_tokens     = n_tok;
    rp.n_tokens_pad = ((n_tok + MASK_PAD - 1) / MASK_PAD) * MASK_PAD;
    rp.n_kv         = seq_len;
    rp.sampling_positions.push_back(sampling_pos);
    rp.gather_idxs.resize(seq_len);
    for (std::int32_t k = 0; k < seq_len; ++k) {
        rp.gather_idxs[k] = static_cast<std::int32_t>(pos_to_phys(k));
    }
    build_causal_mask_f16(rp.mask_f16, seq_len, n_tok, rp.n_tokens_pad,
                          plan.positions_i32.data());
    rp.sampler = SamplerParams{};
    rp.sampler.temperature = 0.0f; // greedy for offline test mode
    plan.sampling_pos_i32.push_back(sampling_pos);
    plan.reqs.push_back(std::move(rp));

    // Detect pure-decode (single token) for the M11 fast path.
    if (n_tok == 1) {
        plan.pure_decode = true;
        plan.max_n_kv = seq_len;
        plan.packed_gather_idxs = plan.reqs[0].gather_idxs;
        const auto zero = ggml_fp32_to_fp16(0.0f);
        plan.packed_mask_f16.assign(
            static_cast<std::size_t>(seq_len) * MASK_PAD, ggml_fp32_to_fp16(-INFINITY));
        for (std::int32_t k = 0; k < seq_len; ++k) {
            plan.packed_mask_f16[k] = zero;
        }
    }
    return plan;
}

// -----------------------------------------------------------------------------
// Compute
// -----------------------------------------------------------------------------

std::vector<SamplerOutput> ForwardEngine::compute_(const BatchPlan& plan) {
    // Tensor metadata + graph node arrays. Sized to match GRAPH_MAX_NODES
    // (the same budget the graph builder uses).
    const std::size_t mem_size =
        ggml_tensor_overhead() * (1ull << 20) +
        ggml_graph_overhead_custom(GRAPH_MAX_NODES, false);
    ggml_init_params ip{
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context* ctx = ggml_init(ip);
    if (!ctx) throw std::runtime_error("compute: ggml_init failed");

    // RAII for the context; ensures ggml_free() runs on every exit path.
    auto ctx_guard = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
        ctx, &ggml_free);

    const GraphResult g = build_qwen3_graph(ctx, model_, kv_, plan);

    if (!ggml_gallocr_alloc_graph(galloc_, g.gf)) {
        throw std::runtime_error("compute: gallocr_alloc_graph failed");
    }

    upload_graph_inputs(g, plan);

    const auto status = ggml_backend_graph_compute(model_.backend(), g.gf);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("compute: graph_compute status=" +
                                 std::to_string(static_cast<int>(status)));
    }

    const std::int32_t vocab_size = model_.hparams().vocab_size;
    const std::int32_t n_slots =
        static_cast<std::int32_t>(plan.sampling_pos_i32.size());
    std::vector<float> all_logits(
        static_cast<std::size_t>(vocab_size) * n_slots);
    ggml_backend_tensor_get(g.logits, all_logits.data(), 0,
                            all_logits.size() * sizeof(float));

    return sample_batch(plan, all_logits.data(), vocab_size);
}

// -----------------------------------------------------------------------------
// Public entry points
// -----------------------------------------------------------------------------

std::size_t ForwardEngine::run(const schema::DecodedRequest& req,
                               std::span<std::uint8_t> response) {
    BatchPlan plan;
    try {
        plan = plan_(req);
    } catch (const std::exception& e) {
        std::cerr << "[forward] plan failed: " << e.what() << "\n";
        return 0;
    }

    std::vector<SamplerOutput> sampled;
    try {
        sampled = compute_(plan);
    } catch (const std::exception& e) {
        std::cerr << "[forward] compute failed: " << e.what() << "\n";
        return 0;
    }

    if (needs_msgpack_mode(sampled)) {
        return write_msgpack_response(response, sampled);
    }
    // Flat fast path — every slot is token-producing. Variable-length
    // per-request tokens (M8 spec decode) are concatenated; the per-
    // request count goes into the counts table.
    std::vector<std::uint32_t> tokens_per_req;
    std::vector<std::uint32_t> tokens;
    tokens_per_req.reserve(sampled.size());
    for (const auto& s : sampled) {
        tokens_per_req.push_back(static_cast<std::uint32_t>(s.tokens.size()));
        tokens.insert(tokens.end(), s.tokens.begin(), s.tokens.end());
    }
    return write_flat_response(response,
                               std::span<const std::uint32_t>(tokens_per_req),
                               std::span<const std::uint32_t>(tokens));
}

std::vector<std::uint32_t> ForwardEngine::generate(
        std::span<const std::uint32_t> prompt_tokens,
        std::int32_t max_new_tokens,
        std::uint64_t /*context_id*/,
        std::int32_t page_offset) {
    if (prompt_tokens.empty()) {
        throw std::runtime_error("generate: empty prompt");
    }
    std::vector<std::uint32_t> out;
    out.reserve(static_cast<std::size_t>(max_new_tokens));

    const std::int32_t prompt_n = static_cast<std::int32_t>(prompt_tokens.size());

    std::vector<std::uint32_t> positions(prompt_tokens.size());
    for (std::size_t i = 0; i < prompt_tokens.size(); ++i) {
        positions[i] = static_cast<std::uint32_t>(i);
    }

    auto plan = plan_test_simple_(prompt_tokens,
                                  std::span<const std::uint32_t>(positions),
                                  prompt_n - 1, page_offset);
    auto sampled = compute_(plan);
    out.push_back(sampled[0].tokens.front());

    for (std::int32_t i = 1; i < max_new_tokens; ++i) {
        const std::uint32_t pos = static_cast<std::uint32_t>(prompt_n + i - 1);
        const std::array<std::uint32_t, 1> tok{out.back()};
        const std::array<std::uint32_t, 1> p{pos};
        auto plan_step = plan_test_simple_(std::span<const std::uint32_t>(tok),
                                           std::span<const std::uint32_t>(p),
                                           /*sampling_pos=*/ 0, page_offset);
        auto step_out = compute_(plan_step);
        out.push_back(step_out[0].tokens.front());
    }
    return out;
}

std::vector<std::vector<std::uint32_t>> ForwardEngine::generate_multi(
        std::vector<std::vector<std::uint32_t>>& prompts,
        std::int32_t max_new_tokens,
        std::vector<std::uint64_t> /*context_ids*/) {
    if (prompts.empty()) {
        throw std::runtime_error("generate_multi: no prompts");
    }
    const std::size_t n_req = prompts.size();
    std::vector<std::vector<std::uint32_t>> out(n_req);
    std::vector<std::int32_t> prompt_lens(n_req);
    std::vector<std::int32_t> page_offsets(n_req);

    // Allocate non-overlapping page ranges per context.
    const std::int32_t page_size = kv_.page_size();
    std::int32_t cursor = 0;
    for (std::size_t r = 0; r < n_req; ++r) {
        const std::int32_t need = static_cast<std::int32_t>(prompts[r].size()) +
                                  max_new_tokens;
        const std::int32_t pages = (need + page_size - 1) / page_size;
        page_offsets[r] = cursor;
        cursor += pages;
        if (cursor > kv_.total_pages()) {
            throw std::runtime_error(
                "generate_multi: not enough pages (need " +
                std::to_string(cursor) + ", have " +
                std::to_string(kv_.total_pages()) + ")");
        }
    }

    // ---- Prefill each context separately (single-request plans). -----------
    for (std::size_t r = 0; r < n_req; ++r) {
        const auto& p = prompts[r];
        if (p.empty()) {
            throw std::runtime_error("generate_multi: empty prompt at " +
                                     std::to_string(r));
        }
        prompt_lens[r] = static_cast<std::int32_t>(p.size());

        std::vector<std::uint32_t> positions(p.size());
        for (std::size_t i = 0; i < p.size(); ++i) {
            positions[i] = static_cast<std::uint32_t>(i);
        }
        auto plan = plan_test_simple_(std::span<const std::uint32_t>(p),
                                      std::span<const std::uint32_t>(positions),
                                      prompt_lens[r] - 1, page_offsets[r]);
        auto sampled = compute_(plan);
        out[r].push_back(sampled[0].tokens.front());
    }

    // ---- Decode loop: ONE multi-request plan per step. ---------------------
    for (std::int32_t step = 1; step < max_new_tokens; ++step) {
        BatchPlan plan;
        plan.total_n_tokens = static_cast<std::int32_t>(n_req);
        plan.tokens_i32.resize(n_req);
        plan.positions_i32.resize(n_req);
        plan.kv_idxs_i64.resize(n_req);
        plan.sampling_pos_i32.resize(n_req);
        plan.reqs.reserve(n_req);

        for (std::size_t r = 0; r < n_req; ++r) {
            const std::int32_t pos = prompt_lens[r] + step - 1;
            const std::int32_t qo_start = static_cast<std::int32_t>(r);
            const std::int32_t page_offset = page_offsets[r];

            auto pos_to_phys = [&](std::int32_t p) -> std::int64_t {
                const std::int32_t page = page_offset + p / page_size;
                return static_cast<std::int64_t>(page) * page_size + (p % page_size);
            };

            plan.tokens_i32[r]    = static_cast<std::int32_t>(out[r].back());
            plan.positions_i32[r] = pos;
            plan.kv_idxs_i64[r]   = pos_to_phys(pos);
            plan.sampling_pos_i32[r] = qo_start;

            const std::int32_t seq_len = pos + 1;
            ReqPlan rp;
            rp.qo_start     = qo_start;
            rp.n_tokens     = 1;
            rp.n_tokens_pad = MASK_PAD;
            rp.n_kv         = seq_len;
            rp.sampling_positions.push_back(qo_start);
            rp.gather_idxs.resize(seq_len);
            for (std::int32_t k = 0; k < seq_len; ++k) {
                rp.gather_idxs[k] = static_cast<std::int32_t>(pos_to_phys(k));
            }
            const std::int32_t one_pos[1] = {pos};
            build_causal_mask_f16(rp.mask_f16, seq_len, 1, MASK_PAD, one_pos);
            rp.sampler.temperature = 0.0f;  // greedy for the test harness
            plan.reqs.push_back(std::move(rp));
        }

        // Activate the M11 packed-decode fast path for this multi-context
        // step (every request has n_tokens=1, no custom attention masks).
        plan.pure_decode = true;
        plan.max_n_kv = 0;
        for (const auto& rp : plan.reqs) plan.max_n_kv = std::max(plan.max_n_kv, rp.n_kv);
        plan.packed_gather_idxs.assign(
            static_cast<std::size_t>(plan.max_n_kv) * n_req, 0);
        plan.packed_mask_f16.assign(
            static_cast<std::size_t>(plan.max_n_kv) * MASK_PAD * n_req,
            ggml_fp32_to_fp16(-INFINITY));
        const auto zero_f16 = ggml_fp32_to_fp16(0.0f);
        for (std::size_t r = 0; r < n_req; ++r) {
            const auto& rp = plan.reqs[r];
            for (std::int32_t k = 0; k < rp.n_kv; ++k) {
                plan.packed_gather_idxs[r * plan.max_n_kv + k] = rp.gather_idxs[k];
            }
            std::uint16_t* mrow = plan.packed_mask_f16.data()
                + static_cast<std::size_t>(r) * plan.max_n_kv * MASK_PAD;
            for (std::int32_t k = 0; k < rp.n_kv; ++k) {
                mrow[k] = zero_f16;
            }
        }

        auto sampled = compute_(plan);
        for (std::size_t r = 0; r < n_req; ++r) {
            out[r].push_back(sampled[r].tokens.front());
        }
    }
    return out;
}

}  // namespace pie_ggml_driver
