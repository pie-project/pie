#include "model/llama_like.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include <cuda_runtime.h>

#include "custom_all_reduce.hpp"
#include "cuda_check.hpp"
#include "kernels/add_bias.hpp"
#include "kernels/embed.hpp"
#include "kernels/gather_rows.hpp"
#include "kernels/head_dim_pad.hpp"
#include "kernels/kv_paged.hpp"
#include "kernels/residual_add.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include "kernels/sample_temp.hpp"
#include "kernels/split_packed.hpp"
#include "kernels/swiglu.hpp"
#include "model/qwen3_vl_vision_forward.hpp"  // scatter_qwen3vl_vision
#include "ops/attention_flashinfer.hpp"
#include "ops/gemm.hpp"

namespace pie_cuda_driver::model {

namespace {

inline void maybe_add_bias(
    void* out, const DeviceTensor* bias_tensor,
    int N, int dim, cudaStream_t stream)
{
    if (bias_tensor == nullptr) return;
    kernels::launch_add_bias_bf16(out, bias_tensor->data(), N, dim, stream);
}

bool decode_full_attention_variant_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_CUDA_DECODE_FULL_ATTENTION");
        if (v == nullptr || v[0] == '\0') return true;
        return v[0] != '0';
    }();
    return enabled;
}

inline void apply_rope(
    const LlamaLikeForwardCfg& fwd_cfg,
    const HfConfig& cfg,
    void* q, void* k,
    const std::int32_t* positions,
    int N, int num_q_heads, int num_kv_heads, int head_dim,
    cudaStream_t stream)
{
    if (fwd_cfg.rope_kind == RopeKind::YaRN) {
        kernels::launch_rope_yarn_bf16(
            q, k, positions,
            N, num_q_heads, num_kv_heads, head_dim,
            cfg.rope_theta,
            fwd_cfg.yarn_factor,
            fwd_cfg.yarn_low_freq_factor,
            fwd_cfg.yarn_high_freq_factor,
            fwd_cfg.yarn_original_max_position,
            stream);
    } else if (fwd_cfg.rope_kind == RopeKind::YaRNOriginal) {
        kernels::launch_rope_yarn_original_bf16(
            q, k, positions,
            N, num_q_heads, num_kv_heads, head_dim,
            cfg.rope_theta,
            fwd_cfg.yarn_factor,
            fwd_cfg.yarn_beta_fast,
            fwd_cfg.yarn_beta_slow,
            fwd_cfg.yarn_attention_factor,
            fwd_cfg.yarn_original_max_position,
            stream);
    } else {
        kernels::launch_rope_bf16(
            q, k, positions,
            N, num_q_heads, num_kv_heads, head_dim,
            cfg.rope_theta, stream);
    }
}

}  // namespace

void prepare_llama_like_decode_plan(
    LlamaLikePlanState& state,
    AttentionWorkspace& attn_ws,
    KvCache& cache,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_h,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_h,
    const std::uint32_t* kv_last_page_lens_d,
    int total_tokens,
    int num_requests,
    bool is_pure_decode)
{
    // The prepare hook runs OUTSIDE any cuStreamCapture region. It updates
    // pinned/device buffers in `attn_ws` that the captured body reads via
    // cudaMemcpyAsync at replay time, so the same captured graph stays
    // correct across fires with different KV lengths.
    state.use_xqa_decode = false;
    state.xqa_max_pages_per_seq = 0;
    state.use_prefill_plan = false;
    state.use_prefill_decode_plan = false;
    if (is_pure_decode && fwd_cfg.use_xqa_decode &&
        cache.format().is_native_bf16() && !cache.hnd_layout()) {
        int max_pages = 1;
        for (int r = 0; r < num_requests; ++r) {
            const int pages = static_cast<int>(
                kv_page_indptr_h[r + 1] - kv_page_indptr_h[r]);
            max_pages = std::max(max_pages, pages);
        }
        state.use_xqa_decode = true;
        state.xqa_max_pages_per_seq =
            ops::xqa_decode_page_bucket(max_pages);
        return;
    }
    if (!is_pure_decode) {
        // Real prefill/mixed batches share one attention schedule across all
        // layers when the model has a single global attention window. Plan it
        // once here so the forward body can dispatch per layer without doing
        // host planner work inside the model loop. Models with alternating
        // sliding-window layouts keep the older per-layer planner path.
        if (fwd_cfg.per_layer_window_left.empty()) {
            if (!state.prefill_plan) {
                state.prefill_plan = ops::make_prefill_plan();
            }
            const int T = (fwd_cfg.tp_size > 0) ? fwd_cfg.tp_size : 1;
            const int num_q_heads_local  = cfg.num_attention_heads / T;
            const int num_kv_heads_local = cfg.num_key_value_heads / T;
            ops::plan_attention_flashinfer_prefill_bf16(
                *state.prefill_plan,
                qo_indptr_h,
                kv_page_indptr_h,
                kv_last_page_lens_h,
                total_tokens,
                num_requests,
                num_q_heads_local,
                num_kv_heads_local,
                cfg.head_dim_kernel,
                cache.page_size(),
                attn_ws,
                /*stream=*/nullptr,
                /*enable_cuda_graph=*/false,
                fwd_cfg.sliding_window,
                /*full_attention_variant=*/false,
                cache.hnd_layout(),
                /*causal_mask=*/true);
            state.use_prefill_plan = true;
        }
        return;
    }
    if (fwd_cfg.force_prefill_path) {
        state.use_prefill_plan = false;
        state.use_prefill_decode_plan = false;
        return;
    }
    const int min_prefill_decode_pages =
        std::max(0, fwd_cfg.prefill_decode_min_kv_pages);
    std::uint64_t total_kv_pages = 0;
    for (int r = 0; r < num_requests; ++r) {
        total_kv_pages += static_cast<std::uint64_t>(
            kv_page_indptr_h[r + 1] - kv_page_indptr_h[r]);
    }
    const int avg_kv_pages = num_requests > 0
        ? static_cast<int>((total_kv_pages + num_requests - 1) / num_requests)
        : 0;
    state.use_prefill_decode_plan =
        fwd_cfg.use_prefill_decode_plan &&
        (min_prefill_decode_pages == 0 ||
         avg_kv_pages >= min_prefill_decode_pages);
    if (state.use_prefill_decode_plan) {
        if (!state.prefill_decode_plan) {
            state.prefill_decode_plan = ops::make_prefill_plan();
        }
        const int T = (fwd_cfg.tp_size > 0) ? fwd_cfg.tp_size : 1;
        const int num_q_heads_local  = cfg.num_attention_heads / T;
        const int num_kv_heads_local = cfg.num_key_value_heads / T;
        auto& qo_indptr_h = state.prefill_decode_qo_indptr_h;
        qo_indptr_h.resize(num_requests + 1);
        for (int r = 0; r <= num_requests; ++r) {
            qo_indptr_h[r] = static_cast<std::uint32_t>(r);
        }
        const int min_full_attention_pages =
            std::max(0, fwd_cfg.prefill_decode_full_attention_min_kv_pages);
        const bool full_attention_variant =
            fwd_cfg.prefill_decode_full_attention_min_requests > 0 &&
            num_requests >= fwd_cfg.prefill_decode_full_attention_min_requests &&
            (min_full_attention_pages == 0 ||
             avg_kv_pages >= min_full_attention_pages) &&
            fwd_cfg.sliding_window < 0 &&
            fwd_cfg.per_layer_window_left.empty();
        ops::plan_attention_flashinfer_prefill_bf16(
            *state.prefill_decode_plan, qo_indptr_h.data(), kv_page_indptr_h,
            kv_last_page_lens_h,
            /*total_tokens=*/num_requests, num_requests,
            num_q_heads_local, num_kv_heads_local, cfg.head_dim_kernel,
            cache.page_size(), attn_ws, /*stream=*/nullptr,
            fwd_cfg.decode_plan_cuda_graph, fwd_cfg.sliding_window,
            full_attention_variant, cache.hnd_layout(),
            /*causal_mask=*/false);
        return;
    }
    if (!state.decode_plan) {
        state.decode_plan = ops::make_decode_plan();
    }
    const int T = (fwd_cfg.tp_size > 0) ? fwd_cfg.tp_size : 1;
    const int num_q_heads_local  = cfg.num_attention_heads / T;
    const int num_kv_heads_local = cfg.num_key_value_heads / T;
    ops::plan_attention_flashinfer_decode(
        *state.decode_plan, kv_page_indptr_h, num_requests,
        num_q_heads_local, num_kv_heads_local, cfg.head_dim_kernel,
        cache.page_size(), attn_ws, /*stream=*/nullptr,
        fwd_cfg.decode_plan_cuda_graph,
        decode_full_attention_variant_enabled() &&
            fwd_cfg.sliding_window < 0 && fwd_cfg.per_layer_window_left.empty(),
        cache.hnd_layout());
}

std::uint32_t llama_like_decode_graph_layout(
    const LlamaLikePlanState& state)
{
    if (state.use_xqa_decode) {
        return ops::xqa_decode_graph_layout(state.xqa_max_pages_per_seq);
    }
    if (state.use_prefill_decode_plan && state.prefill_decode_plan) {
        return ops::prefill_plan_graph_layout(*state.prefill_decode_plan);
    }
    if (!state.decode_plan) return 0;
    return ops::decode_plan_graph_layout(*state.decode_plan);
}

void llama_like_forward_paged(
    const Qwen3Weights& w,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    const LlamaLikePlanState& plan_state,
    Qwen3Workspace& ws,
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
    const std::uint32_t* kv_page_indptr_h,
    int N,
    int R,
    bool is_pure_decode,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows,
    bool tp_greedy_argmax,
    const std::uint8_t* custom_mask_d,
    const std::int32_t* custom_mask_indptr_d,
    const LlamaLikeVisionInputs* vision)
{
    // Tensor-parallel local dims. tp_size == 1 reverts to single-GPU
    // shapes; the local *_local fields just shadow the unsharded value.
    // Every tp-aware dim (q/k heads, intermediate, lm_head bound to embed)
    // must divide cleanly by tp_size — checked at engine load.
    const int T  = (fwd_cfg.tp_size > 0) ? fwd_cfg.tp_size : 1;
    const int H  = cfg.hidden_size;
    const int Hq_full = cfg.num_attention_heads * cfg.head_dim;
    const int Hk_full = cfg.num_key_value_heads * cfg.head_dim;
    const int I_full  = cfg.intermediate_size;
    const int Hq = Hq_full / T;
    const int Hk = Hk_full / T;
    const int I  = I_full / T;
    const int num_q_heads_local  = cfg.num_attention_heads / T;
    const int num_kv_heads_local = cfg.num_key_value_heads / T;
    const int V  = cfg.vocab_size;
    const int d  = cfg.head_dim;
    const int dk = cfg.head_dim_kernel;            // padded HEAD_DIM the
                                                   // attention kernel runs at
    const bool head_dim_padded = (d != dk);
    const float eps = cfg.rms_norm_eps;
    // Inherit the stream bound to `cublas` so manual kernel launches stay
    // on the same stream as the cublas matmuls. The graph-capture path
    // in executor/executor.cpp binds `cublas` to its `cstream` for the
    // duration of capture so every launch in this body — cublas-issued or
    // not — lands on the captured graph.
    cudaStream_t stream = cublas.stream();
    NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;
    const bool native_bf16_kv_cache = cache.format().is_native_bf16();

    // When head_dim is padded, the attention kernel runs at `dk`
    // (e.g. 128) and consumes Q/K/V from the *padded* workspace
    // buffers; the packed buffers stay at `d` (e.g. 96) for the
    // GEMM in/out paths. flashinfer's softmax expects
    // `1/sqrt(head_dim)`; with padding we need `1/sqrt(d)` (the
    // real head_dim) regardless of `dk`.
    const float sm_scale_override = head_dim_padded
        ? (1.0f / std::sqrt(static_cast<float>(d)))
        : -1.f;  // -1 = let the dispatch pick `1/sqrt(dk)`

    // 1. Embed.
    kernels::launch_embed_bf16(
        token_ids, w.embed->data(), ws.y.data(),
        N, H, cfg.vocab_size, stream);

    // 1b. Qwen3-VL multimodal: encode each image and overwrite its soft-token
    // rows in the embed output; also stash the deepstack merger outputs (added
    // into the hidden state on image rows after early decoder layers below).
    if (vision != nullptr && vision->vision_in != nullptr &&
        vision->vision_in->num_images > 0) {
        scatter_qwen3vl_vision(
            *vision->vision_in,
            static_cast<__nv_bfloat16*>(ws.y.data()),
            N, H,
            static_cast<__nv_bfloat16*>(vision->deepstack_scratch),
            vision->num_deepstack, cublas.handle(), stream);
    }

    // Some GQA group sizes (Qwen2 small models — 6, 7) aren't in
    // flashinfer's decode dispatch table; for those we run the prefill
    // kernel even for qo_len==1 batches. The runtime decision lives in
    // a single bool: anything past the plan_attention call uses it.
    const bool use_xqa_decode_path =
        is_pure_decode && plan_state.use_xqa_decode;
    const bool use_decode_path =
        is_pure_decode && (!fwd_cfg.force_prefill_path || use_xqa_decode_path);
    const bool use_prefill_decode_path =
        use_decode_path && !use_xqa_decode_path &&
        plan_state.use_prefill_decode_plan;

    // Decode plan was set up by the prepare hook (runs outside any
    // cudaStream capture region) — the body just reads from
    // `plan_state.decode_plan`. Keeps the body purely device-side so
    // the executor can capture it into a CUDA graph.
    const ops::DecodePlanCache* decode_plan =
        plan_state.decode_plan ? plan_state.decode_plan.get() : nullptr;
    const ops::PrefillPlanCache* prefill_decode_plan =
        plan_state.prefill_decode_plan ? plan_state.prefill_decode_plan.get() : nullptr;
    const ops::PrefillPlanCache* prefill_plan =
        plan_state.prefill_plan ? plan_state.prefill_plan.get() : nullptr;
    if (use_xqa_decode_path) {
        ops::prepare_attention_xqa_decode_bf16(
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens,
            R,
            cache.page_size(),
            plan_state.xqa_max_pages_per_seq,
            attn_ws,
            stream);
    }

    const bool post_norm = fwd_cfg.norm_placement == NormPlacement::Post;
    bool have_next_attn_norm = false;
    bool have_final_norm = false;
    const void* final_norm_buf = nullptr;
    bool rope_table_ready = false;
    const float* rope_table = nullptr;

    for (int L = 0; L < cfg.num_hidden_layers; ++L) {
        const auto& layer = w.layers[L];

        // Pre-norm: norm(y) → norm_x; QKV reads from norm_x.
        // Post-norm (OLMo-3): QKV reads y directly; norm_attn applied
        //                     to attn_out *before* the residual add.
        const void* qkv_in = ws.y.data();
        if (!post_norm) {
            if (!have_next_attn_norm) {
                kernels::launch_rmsnorm_bf16(
                    ws.y.data(), layer.attn_norm->data(), ws.norm_x.data(),
                    N, H, eps, stream);
            }
            qkv_in = ws.norm_x.data();
            have_next_attn_norm = false;
        }

        // QKV: fused path when the bind helper materialised
        // `qkv_proj_fused` (single wide gemm + split kernel), unfused
        // fallback for quantized projections / TP-sharded loads /
        // architectures that haven't opted in yet.
        const bool q_norm_is_per_head =
            layer.q_norm && layer.q_norm->shape().size() == 1 &&
            layer.q_norm->shape()[0] == d;
        const bool k_norm_is_per_head =
            layer.k_norm && layer.k_norm->shape().size() == 1 &&
            layer.k_norm->shape()[0] == d;
        const bool use_fused_qkv = (layer.qkv_proj_fused != nullptr) &&
                                   !ws.qkv_fused.empty();
        const bool fused_decode_qkv_post =
            use_fused_qkv &&
            is_pure_decode &&
            native_bf16_kv_cache &&
            !head_dim_padded &&
            !fwd_cfg.use_qkv_bias &&
            fwd_cfg.use_qk_norm &&
            q_norm_is_per_head && k_norm_is_per_head &&
            fwd_cfg.rope_kind == RopeKind::Standard;
        if (use_fused_qkv) {
            ops::gemm_act_x_w(cublas.handle(),
                qkv_in, ops::WeightView(*layer.qkv_proj_fused),
                ws.qkv_fused.data(), N, Hq + 2 * Hk, H);
            if (fused_decode_qkv_post) {
                if (!rope_table_ready && !ws.rope_table.empty()) {
                    kernels::launch_rope_standard_table(
                        positions,
                        static_cast<float*>(ws.rope_table.data()),
                        N, d, cfg.rope_theta, stream);
                    rope_table =
                        static_cast<const float*>(ws.rope_table.data());
                    rope_table_ready = true;
                }
                kernels::launch_qkv_decode_qk_norm_rope_write_kv_bf16(
                    ws.qkv_fused.data(),
                    ws.q.data(),
                    cache.k(L), cache.v(L),
                    layer.q_norm->data(), layer.k_norm->data(),
                    positions,
                    rope_table,
                    kv_page_indices, kv_page_indptr, kv_last_page_lens,
                    R, num_q_heads_local, num_kv_heads_local, d,
                    cache.page_size(), cache.hnd_layout(),
                    cfg.rope_theta, eps, stream);
            } else {
                kernels::launch_split_qkv_bf16(
                    ws.qkv_fused.data(),
                    ws.q.data(), ws.k.data(), ws.v.data(),
                    N, Hq, Hk, stream);
            }
        } else {
            ops::gemm_act_x_w(cublas.handle(),
                qkv_in, make_weight_view(layer.q_proj, layer.q_proj_quant),
                ws.q.data(), N, Hq, H);
            ops::gemm_act_x_w(cublas.handle(),
                qkv_in, make_weight_view(layer.k_proj, layer.k_proj_quant),
                ws.k.data(), N, Hk, H);
            ops::gemm_act_x_w(cublas.handle(),
                qkv_in, make_weight_view(layer.v_proj, layer.v_proj_quant),
                ws.v.data(), N, Hk, H);
        }

        if (!fused_decode_qkv_post && fwd_cfg.use_qkv_bias) {
            maybe_add_bias(ws.q.data(), layer.q_bias, N, Hq, stream);
            maybe_add_bias(ws.k.data(), layer.k_bias, N, Hk, stream);
            maybe_add_bias(ws.v.data(), layer.v_bias, N, Hk, stream);
        }

        // q_norm / k_norm: two conventions ship in the wild.
        //   * Per-head (Qwen3, OLMo-2 small, Gemma-3): weight shape
        //     `[head_dim]`. RMSNorm rolls each head's `d` channels
        //     independently — `num_rows = N*num_heads`, `hidden = d`.
        //   * Global (OLMo-2 7B+, OLMo-3): weight shape `[H_*]`
        //     (per-rank in TP). HF flattens [N, num_heads, d] → [N, H_*]
        //     and applies one RMSNorm with the full vector — `num_rows
        //     = N`, `hidden = num_heads * d`. Pre-rope behaviour
        //     differs from per-head by the shared scale across heads.
        // Dispatch by inspecting the bound q/k_norm shape; bind code
        // doesn't reshape, so the tensor's leading dim tells us.
        auto rmsnorm_qk = [&](void* x, const DeviceTensor* w,
                              int num_heads_local, int per_rank_H) {
            const bool global_norm = (w->shape().size() == 1 &&
                                      w->shape()[0] == per_rank_H);
            if (global_norm) {
                kernels::launch_rmsnorm_bf16(
                    x, w->data(), x, N, per_rank_H, eps, stream);
            } else {
                kernels::launch_rmsnorm_bf16(
                    x, w->data(), x,
                    N * num_heads_local, d, eps, stream);
            }
        };
        const bool fuse_qk_norm_rope =
            fwd_cfg.use_qk_norm &&
            q_norm_is_per_head && k_norm_is_per_head &&
            fwd_cfg.rope_kind == RopeKind::Standard;
        const bool use_mrope =
            fwd_cfg.rope_kind == RopeKind::MRopeInterleaved &&
            vision != nullptr && vision->mrope_positions != nullptr &&
            q_norm_is_per_head && k_norm_is_per_head;
        if (fused_decode_qkv_post) {
            // Q was normalized/rotated and K/V were written directly to the
            // paged cache by the fused decode postprocess above.
        } else if (use_mrope) {
            // Qwen3-VL: fused per-head q/k RMSNorm + interleaved 3-axis M-RoPE.
            kernels::launch_qk_rmsnorm_mrope_bf16(
                ws.q.data(), ws.k.data(),
                layer.q_norm->data(), layer.k_norm->data(),
                vision->mrope_positions,
                N, num_q_heads_local, num_kv_heads_local, d,
                cfg.rope_theta, eps,
                fwd_cfg.mrope_section_t, fwd_cfg.mrope_section_h,
                fwd_cfg.mrope_section_w, stream);
        } else if (fuse_qk_norm_rope) {
            kernels::launch_qk_rmsnorm_rope_bf16(
                ws.q.data(), ws.k.data(),
                layer.q_norm->data(), layer.k_norm->data(),
                positions, N, num_q_heads_local, num_kv_heads_local, d,
                cfg.rope_theta, eps, stream);
        } else {
            if (fwd_cfg.use_qk_norm && layer.q_norm) {
                rmsnorm_qk(ws.q.data(), layer.q_norm, num_q_heads_local, Hq);
            }
            if (fwd_cfg.use_qk_norm && layer.k_norm) {
                rmsnorm_qk(ws.k.data(), layer.k_norm, num_kv_heads_local, Hk);
            }

            apply_rope(fwd_cfg, cfg,
                       ws.q.data(), ws.k.data(), positions,
                       N, num_q_heads_local, num_kv_heads_local, d,
                       stream);
        }

        // Pad Q/K/V to `dk` when the model's head_dim isn't a flashinfer
        // dispatch value. The padded buffers are zero on the trailing
        // `dk - d` cols per head; QK·V dot products therefore equal the
        // unpadded ones, and `sm_scale = 1/sqrt(d)` keeps the softmax
        // scaled to the real head dim.
        const void* attn_q   = ws.q.data();
        const void* attn_k   = ws.k.data();
        const void* attn_v   = ws.v.data();
        void* attn_out_buf   = ws.attn_out.data();
        if (!fused_decode_qkv_post && head_dim_padded) {
            kernels::launch_pad_head_dim_bf16(
                ws.q.data(), ws.q_padded.data(),
                N, num_q_heads_local, d, dk, stream);
            kernels::launch_pad_head_dim_bf16(
                ws.k.data(), ws.k_padded.data(),
                N, num_kv_heads_local, d, dk, stream);
            kernels::launch_pad_head_dim_bf16(
                ws.v.data(), ws.v_padded.data(),
                N, num_kv_heads_local, d, dk, stream);
            attn_q = ws.q_padded.data();
            attn_k = ws.k_padded.data();
            attn_v = ws.v_padded.data();
            attn_out_buf = ws.attn_out_padded.data();
        }

        auto kv_view = cache.layer_view(L);
        if (fused_decode_qkv_post) {
            // Already written by launch_qkv_decode_qk_norm_rope_write_kv_bf16.
        } else {
            kernels::launch_write_kv_to_pages(
                kv_view,
                const_cast<void*>(attn_k), const_cast<void*>(attn_v),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                N, R, stream);
        }

        // Per-layer sliding-window dispatch (OLMo-3, Mistral). When
        // `per_layer_window_left` is empty we fall back to the global
        // `sliding_window`; that single value is broadcast to every
        // layer (used by Mistral / Gemma-2 single-mode and Phi-3).
        const int layer_window_left =
            (!fwd_cfg.per_layer_window_left.empty() &&
             L < static_cast<int>(fwd_cfg.per_layer_window_left.size()))
                ? fwd_cfg.per_layer_window_left[L]
                : fwd_cfg.sliding_window;

        if (use_xqa_decode_path) {
            ops::launch_attention_xqa_decode_bf16_prepared(
                attn_q, kv_view.k_bf16_pages, kv_view.v_bf16_pages, attn_out_buf,
                R, num_q_heads_local, num_kv_heads_local, dk,
                cache.page_size(), plan_state.xqa_max_pages_per_seq,
                attn_ws, stream, sm_scale_override);
        } else if (use_prefill_decode_path) {
            const int num_pages_in_batch = kv_page_indptr_h[R];
            kernels::launch_dequant_kv_cache_layer_to_bf16_active(
                kv_view, kv_page_indices, num_pages_in_batch, stream);
            ops::dispatch_attention_flashinfer_prefill_bf16(
                *prefill_decode_plan,
                attn_q, kv_view.k_bf16_pages, kv_view.v_bf16_pages, attn_out_buf,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                attn_ws, stream, /*logits_soft_cap=*/0.f, sm_scale_override);
        } else if (use_decode_path) {
            ops::dispatch_attention_flashinfer_decode(
                *decode_plan,
                attn_q, kv_view, attn_out_buf,
                kv_page_indices, kv_page_indptr, kv_last_page_lens,
                attn_ws, stream, layer_window_left,
                /*logits_soft_cap=*/0.f, sm_scale_override);
        } else if (custom_mask_d) {
            ops::launch_attention_flashinfer_prefill_custom(
                attn_q, kv_view, attn_out_buf,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                custom_mask_d, custom_mask_indptr_d,
                qo_indptr_h, kv_page_indptr_h,
                N, R, num_q_heads_local, attn_ws, stream);
        } else if (plan_state.use_prefill_plan && prefill_plan != nullptr) {
            const int num_pages_in_batch = kv_page_indptr_h[R];
            kernels::launch_dequant_kv_cache_layer_to_bf16_active(
                kv_view, kv_page_indices, num_pages_in_batch, stream);
            ops::dispatch_attention_flashinfer_prefill_bf16(
                *prefill_plan,
                attn_q, kv_view.k_bf16_pages, kv_view.v_bf16_pages, attn_out_buf,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                attn_ws, stream, /*logits_soft_cap=*/0.f, sm_scale_override);
        } else {
            ops::launch_attention_flashinfer_prefill(
                attn_q, kv_view, attn_out_buf,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, num_q_heads_local, attn_ws, stream, layer_window_left,
                /*logits_soft_cap=*/0.f, sm_scale_override);
        }

        // Strip the trailing pad cols off the attention output before
        // it feeds the o_proj GEMM (which expects `[N, num_q*head_dim]`,
        // not `[N, num_q*head_dim_kernel]`).
        if (head_dim_padded) {
            kernels::launch_strip_head_dim_bf16(
                attn_out_buf, ws.attn_out.data(),
                N, num_q_heads_local, d, dk, stream);
        }

        bool have_mlp_norm = false;
        if (!post_norm) {
            // o_proj is row-parallel: each rank's GEMM produces a partial
            // [N, H] contribution. Single-GPU fuses it into y as a
            // residual-add (beta=1); under TP we go via a scratch
            // (ws.norm_x is free here — it held the QKV input before),
            // all-reduce the partials, then add to the residual.
            if (T == 1) {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.attn_out.data(), make_weight_view(layer.o_proj, layer.o_proj_quant),
                    ws.y.data(), N, H, Hq, /*beta=*/1.f);
            } else {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.attn_out.data(), make_weight_view(layer.o_proj, layer.o_proj_quant),
                    ws.norm_x.data(), N, H, Hq, /*beta=*/0.f);
                auto* fused_ar = tp->custom_all_reduce();
                if (fused_ar != nullptr &&
                    fused_ar->can_fuse_residual_rmsnorm(N, H, stream)) {
                    fused_ar->all_reduce_residual_rmsnorm_bf16(
                        ws.norm_x.data(), ws.y.data(), layer.mlp_norm->data(),
                        ws.norm_y.data(), N, H, eps, stream);
                } else {
                    tp->all_reduce_bf16_out(ws.norm_x.data(), ws.norm_y.data(),
                        static_cast<std::size_t>(N) * H, ncclSum, stream);
                    kernels::launch_residual_add_rmsnorm_bf16(
                        ws.y.data(), ws.norm_y.data(), layer.mlp_norm->data(),
                        ws.norm_y.data(), N, H, eps, stream);
                }
                have_mlp_norm = true;
            }
        } else {
            // Post-norm: o_proj writes to norm_x (a scratch we own here),
            // norm_attn(norm_x) → norm_y, then y += norm_y.
            ops::gemm_act_x_w(cublas.handle(),
                ws.attn_out.data(), make_weight_view(layer.o_proj, layer.o_proj_quant),
                ws.norm_x.data(), N, H, Hq, /*beta=*/0.f);
            if (T > 1) {
                tp->all_reduce_bf16(ws.norm_x.data(),
                    static_cast<std::size_t>(N) * H, ncclSum, stream);
            }
            kernels::launch_rmsnorm_bf16(
                ws.norm_x.data(), layer.attn_norm->data(), ws.norm_y.data(),
                N, H, eps, stream);
            kernels::launch_residual_add_bf16(
                ws.y.data(), ws.norm_y.data(), N * H, stream);
        }

        // MLP block.
        const void* mlp_in = ws.y.data();
        if (!post_norm) {
            if (!have_mlp_norm) {
                kernels::launch_rmsnorm_bf16(
                    ws.y.data(), layer.mlp_norm->data(), ws.norm_y.data(),
                    N, H, eps, stream);
            }
            mlp_in = ws.norm_y.data();
        }

        // gate + up: same fused-vs-unfused dispatch as QKV.
        const bool use_fused_gu = (layer.gate_up_proj_fused != nullptr) &&
                                  !ws.gate_up_fused.empty();
        if (use_fused_gu) {
            ops::gemm_act_x_w(cublas.handle(),
                mlp_in, ops::WeightView(*layer.gate_up_proj_fused),
                ws.gate_up_fused.data(), N, 2 * I, H);
            kernels::launch_chunked_swiglu_bf16(
                ws.gate_up_fused.data(), ws.gate.data(), N, I, stream);
        } else {
            ops::gemm_act_x_w(cublas.handle(),
                mlp_in, make_weight_view(layer.gate_proj, layer.gate_proj_quant),
                ws.gate.data(), N, I, H);
            ops::gemm_act_x_w(cublas.handle(),
                mlp_in, make_weight_view(layer.up_proj, layer.up_proj_quant),
                ws.up.data(),   N, I, H);
            kernels::launch_swiglu_bf16(
                ws.gate.data(), ws.up.data(), ws.gate.data(),
                N * I, stream);
        }

        if (!post_norm) {
            // down_proj is row-parallel: same all-reduce + residual-add
            // dance as o_proj. ws.norm_x is free here (it last held the
            // mlp pre-norm input on the post-norm path; on pre-norm it
            // hasn't been touched since QKV).
            if (T == 1) {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.gate.data(), make_weight_view(layer.down_proj, layer.down_proj_quant),
                    ws.y.data(), N, H, I, /*beta=*/1.f);
            } else {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.gate.data(), make_weight_view(layer.down_proj, layer.down_proj_quant),
                    ws.norm_x.data(), N, H, I, /*beta=*/0.f);
                auto* fused_ar = tp->custom_all_reduce();
                if (fused_ar != nullptr &&
                    fused_ar->can_fuse_residual_rmsnorm(N, H, stream)) {
                    if (L + 1 < cfg.num_hidden_layers) {
                        fused_ar->all_reduce_residual_rmsnorm_bf16(
                            ws.norm_x.data(), ws.y.data(),
                            w.layers[L + 1].attn_norm->data(),
                            ws.norm_x.data(), N, H, eps, stream);
                        have_next_attn_norm = true;
                    } else {
                        fused_ar->all_reduce_residual_rmsnorm_bf16(
                            ws.norm_x.data(), ws.y.data(),
                            w.final_norm->data(),
                            ws.norm_y.data(), N, H, eps, stream);
                        have_final_norm = true;
                        final_norm_buf = ws.norm_y.data();
                    }
                } else {
                    tp->all_reduce_bf16_out(ws.norm_x.data(), ws.norm_y.data(),
                        static_cast<std::size_t>(N) * H, ncclSum, stream);
                    kernels::launch_residual_add_bf16(
                        ws.y.data(), ws.norm_y.data(), N * H, stream);
                }
            }
        } else {
            // Post-norm MLP: down_proj → norm_x scratch, norm_mlp, += y.
            ops::gemm_act_x_w(cublas.handle(),
                ws.gate.data(), make_weight_view(layer.down_proj, layer.down_proj_quant),
                ws.norm_x.data(), N, H, I, /*beta=*/0.f);
            if (T > 1) {
                tp->all_reduce_bf16(ws.norm_x.data(),
                    static_cast<std::size_t>(N) * H, ncclSum, stream);
            }
            kernels::launch_rmsnorm_bf16(
                ws.norm_x.data(), layer.mlp_norm->data(), ws.norm_y.data(),
                N, H, eps, stream);
            kernels::launch_residual_add_bf16(
                ws.y.data(), ws.norm_y.data(), N * H, stream);
        }

        // Qwen3-VL DeepStack: after decoder layers 0..num_deepstack-1, add the
        // matching deepstack merger output into the hidden state. The scatter
        // zeroed the scratch and wrote only image rows, so a whole-tensor
        // residual-add leaves text rows unchanged (HF `_deepstack_process`).
        if (vision != nullptr && vision->deepstack_scratch != nullptr &&
            L < vision->num_deepstack) {
            const auto* ds = static_cast<const std::uint16_t*>(
                                 vision->deepstack_scratch) +
                             static_cast<std::size_t>(L) * N * H;
            kernels::launch_residual_add_bf16(
                ws.y.data(), ds, static_cast<std::size_t>(N) * H, stream);
        }
    }

    const bool use_tp_greedy =
        tp_greedy_argmax && T > 1 && tp != nullptr &&
        w.lm_head_tp_shard != nullptr &&
        w.lm_head_tp_shard->shape().size() == 2 &&
        w.lm_head_tp_shard->shape()[0] > 0 &&
        T <= 8;
    if (use_tp_greedy) {
        const bool compact_logits =
            logit_row_indices_d != nullptr && num_logit_rows > 0 &&
            num_logit_rows < N;
        const int V_local = static_cast<int>(w.lm_head_tp_shard->shape()[0]);
        const void* final_act = final_norm_buf;
        int lm_head_rows = N;
        if (compact_logits) {
            if (have_final_norm) {
                kernels::launch_gather_bf16_rows(
                    static_cast<const std::uint16_t*>(final_norm_buf),
                    logit_row_indices_d,
                    static_cast<std::uint16_t*>(ws.norm_x.data()),
                    num_logit_rows, H, stream);
                final_act = ws.norm_x.data();
            } else {
                kernels::launch_gather_bf16_rows(
                    static_cast<const std::uint16_t*>(ws.y.data()),
                    logit_row_indices_d,
                    static_cast<std::uint16_t*>(ws.norm_x.data()),
                    num_logit_rows, H, stream);
                kernels::launch_rmsnorm_bf16(
                    ws.norm_x.data(), w.final_norm->data(), ws.norm_y.data(),
                    num_logit_rows, H, eps, stream);
                final_act = ws.norm_y.data();
            }
            lm_head_rows = num_logit_rows;
        } else if (!have_final_norm) {
            kernels::launch_rmsnorm_bf16(
                ws.y.data(), w.final_norm->data(), ws.norm_x.data(),
                N, H, eps, stream);
            final_act = ws.norm_x.data();
        }
        ops::gemm_act_x_w(cublas.handle(),
            final_act, *w.lm_head_tp_shard, ws.logits.data(),
            lm_head_rows, V_local, H);
        kernels::launch_argmax_bf16_pair_with_offset(
            ws.logits.data(),
            reinterpret_cast<std::uint64_t*>(ws.greedy_pairs.data()),
            lm_head_rows, V_local, w.lm_head_tp_vocab_offset, stream);
        tp->all_gather_bytes(
            ws.greedy_pairs.data(), ws.greedy_pairs_all.data(),
            static_cast<std::size_t>(lm_head_rows) * sizeof(std::uint64_t),
            stream);
    } else if (fwd_cfg.emit_logits) {
        const bool compact_logits =
            logit_row_indices_d != nullptr && num_logit_rows > 0 &&
            num_logit_rows < N;
        const void* lm_head_input =
            have_final_norm ? final_norm_buf : ws.norm_y.data();
        int lm_head_rows = N;
        if (compact_logits) {
            if (have_final_norm) {
                kernels::launch_gather_bf16_rows(
                    static_cast<const std::uint16_t*>(final_norm_buf),
                    logit_row_indices_d,
                    static_cast<std::uint16_t*>(ws.norm_x.data()),
                    num_logit_rows, H, stream);
                lm_head_input = ws.norm_x.data();
            } else {
                kernels::launch_gather_bf16_rows(
                    static_cast<const std::uint16_t*>(ws.y.data()),
                    logit_row_indices_d,
                    static_cast<std::uint16_t*>(ws.norm_x.data()),
                    num_logit_rows, H, stream);
                kernels::launch_rmsnorm_bf16(
                    ws.norm_x.data(), w.final_norm->data(), ws.norm_y.data(),
                    num_logit_rows, H, eps, stream);
                lm_head_input = ws.norm_y.data();
            }
            lm_head_rows = num_logit_rows;
        } else if (!have_final_norm) {
            kernels::launch_rmsnorm_bf16(
                ws.y.data(), w.final_norm->data(), ws.norm_y.data(),
                N, H, eps, stream);
            lm_head_input = ws.norm_y.data();
        }
        ops::gemm_act_x_w(cublas.handle(),
            lm_head_input, *w.lm_head, ws.logits.data(),
            lm_head_rows, V, H);
    }
}

RopeKind rope_kind_from_hf_config(const HfConfig& hf) {
    using RopeScaling = HfConfig::RopeScaling;
    switch (hf.rope_scaling_kind) {
    case RopeScaling::Llama3:
        return RopeKind::YaRN;
    case RopeScaling::OriginalYaRN:
        return RopeKind::YaRNOriginal;
    case RopeScaling::None:
        return RopeKind::Standard;
    }
    return RopeKind::Standard;
}

void apply_rope_config(LlamaLikeForwardCfg& fwd_cfg, const HfConfig& hf) {
    fwd_cfg.rope_kind                  = rope_kind_from_hf_config(hf);
    fwd_cfg.yarn_factor                = hf.rope_factor;
    fwd_cfg.yarn_low_freq_factor       = hf.rope_low_freq_factor;
    fwd_cfg.yarn_high_freq_factor      = hf.rope_high_freq_factor;
    fwd_cfg.yarn_original_max_position = hf.rope_original_max_position;
    fwd_cfg.yarn_beta_fast             = hf.rope_beta_fast;
    fwd_cfg.yarn_beta_slow             = hf.rope_beta_slow;
    fwd_cfg.yarn_attention_factor      = hf.rope_attention_factor;
}

}  // namespace pie_cuda_driver::model
