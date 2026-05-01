#include "model/llama_like.hpp"

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/add_bias.hpp"
#include "kernels/embed.hpp"
#include "kernels/head_dim_pad.hpp"
#include "kernels/kv_paged.hpp"
#include "kernels/residual_add.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include "kernels/swiglu.hpp"
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
    } else {
        kernels::launch_rope_bf16(
            q, k, positions,
            N, num_q_heads, num_kv_heads, head_dim,
            cfg.rope_theta, stream);
    }
}

}  // namespace

void llama_like_forward_paged(
    const Qwen3Weights& w,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
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
    const std::uint8_t* custom_mask_d,
    const std::int32_t* custom_mask_indptr_d)
{
    const int H  = cfg.hidden_size;
    const int Hq = cfg.num_attention_heads * cfg.head_dim;
    const int Hk = cfg.num_key_value_heads * cfg.head_dim;
    const int I  = cfg.intermediate_size;
    const int V  = cfg.vocab_size;
    const int d  = cfg.head_dim;
    const int dk = cfg.head_dim_kernel;            // padded HEAD_DIM the
                                                   // attention kernel runs at
    const int Hq_kern = cfg.num_attention_heads * dk;
    const int Hk_kern = cfg.num_key_value_heads * dk;
    const bool head_dim_padded = (d != dk);
    const float eps = cfg.rms_norm_eps;
    cudaStream_t stream = nullptr;

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

    // Some GQA group sizes (Qwen2 small models — 6, 7) aren't in
    // flashinfer's decode dispatch table; for those we run the prefill
    // kernel even for qo_len==1 batches. The runtime decision lives in
    // a single bool: anything past the plan_attention call uses it.
    const bool use_decode_path = is_pure_decode && !fwd_cfg.force_prefill_path;

    // Hoist decode plan once per fire (matches qwen3_forward). Plan
    // runs at the kernel head_dim (`dk`) so the work estimator picks
    // the right shared-memory footprint for the actual kernel launch.
    ops::DecodePlanCachePtr decode_plan;
    if (use_decode_path) {
        decode_plan = ops::make_decode_plan();
        ops::plan_attention_flashinfer_decode_bf16(
            *decode_plan, kv_page_indptr_h, R,
            cfg.num_attention_heads, cfg.num_key_value_heads, dk,
            cache.page_size(), attn_ws, stream);
    }

    const bool post_norm = fwd_cfg.norm_placement == NormPlacement::Post;

    for (int L = 0; L < cfg.num_hidden_layers; ++L) {
        const auto& layer = w.layers[L];

        // Pre-norm: norm(y) → norm_x; QKV reads from norm_x.
        // Post-norm (OLMo-3): QKV reads y directly; norm_attn applied
        //                     to attn_out *before* the residual add.
        const void* qkv_in = ws.y.data();
        if (!post_norm) {
            kernels::launch_rmsnorm_bf16(
                ws.y.data(), layer.attn_norm->data(), ws.norm_x.data(),
                N, H, eps, stream);
            qkv_in = ws.norm_x.data();
        }

        ops::gemm_act_x_wt_bf16(cublas.handle(),
            qkv_in, layer.q_proj->data(), ws.q.data(), N, Hq, H);
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            qkv_in, layer.k_proj->data(), ws.k.data(), N, Hk, H);
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            qkv_in, layer.v_proj->data(), ws.v.data(), N, Hk, H);

        if (fwd_cfg.use_qkv_bias) {
            maybe_add_bias(ws.q.data(), layer.q_bias, N, Hq, stream);
            maybe_add_bias(ws.k.data(), layer.k_bias, N, Hk, stream);
            maybe_add_bias(ws.v.data(), layer.v_bias, N, Hk, stream);
        }

        if (fwd_cfg.use_qk_norm && layer.q_norm) {
            kernels::launch_rmsnorm_bf16(
                ws.q.data(), layer.q_norm->data(), ws.q.data(),
                N * cfg.num_attention_heads, d, eps, stream);
        }
        if (fwd_cfg.use_qk_norm && layer.k_norm) {
            kernels::launch_rmsnorm_bf16(
                ws.k.data(), layer.k_norm->data(), ws.k.data(),
                N * cfg.num_key_value_heads, d, eps, stream);
        }

        apply_rope(fwd_cfg, cfg,
                   ws.q.data(), ws.k.data(), positions,
                   N, cfg.num_attention_heads, cfg.num_key_value_heads, d,
                   stream);

        // Pad Q/K/V to `dk` when the model's head_dim isn't a flashinfer
        // dispatch value. The padded buffers are zero on the trailing
        // `dk - d` cols per head; QK·V dot products therefore equal the
        // unpadded ones, and `sm_scale = 1/sqrt(d)` keeps the softmax
        // scaled to the real head dim.
        const void* attn_q   = ws.q.data();
        const void* attn_k   = ws.k.data();
        const void* attn_v   = ws.v.data();
        void* attn_out_buf   = ws.attn_out.data();
        if (head_dim_padded) {
            kernels::launch_pad_head_dim_bf16(
                ws.q.data(), ws.q_padded.data(),
                N, cfg.num_attention_heads, d, dk, stream);
            kernels::launch_pad_head_dim_bf16(
                ws.k.data(), ws.k_padded.data(),
                N, cfg.num_key_value_heads, d, dk, stream);
            kernels::launch_pad_head_dim_bf16(
                ws.v.data(), ws.v_padded.data(),
                N, cfg.num_key_value_heads, d, dk, stream);
            attn_q = ws.q_padded.data();
            attn_k = ws.k_padded.data();
            attn_v = ws.v_padded.data();
            attn_out_buf = ws.attn_out_padded.data();
        }

        kernels::launch_write_kv_to_pages_bf16(
            cache.k(L), cache.v(L),
            const_cast<void*>(attn_k), const_cast<void*>(attn_v),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            N, R, cache.page_size(), cfg.num_key_value_heads, dk, stream);

        // Per-layer sliding-window dispatch (OLMo-3, Mistral). When
        // `per_layer_window_left` is empty we fall back to the global
        // `sliding_window`; that single value is broadcast to every
        // layer (used by Mistral / Gemma-2 single-mode and Phi-3).
        const int layer_window_left =
            (!fwd_cfg.per_layer_window_left.empty() &&
             L < static_cast<int>(fwd_cfg.per_layer_window_left.size()))
                ? fwd_cfg.per_layer_window_left[L]
                : fwd_cfg.sliding_window;

        if (use_decode_path) {
            ops::dispatch_attention_flashinfer_decode_bf16(
                *decode_plan,
                attn_q, cache.k(L), cache.v(L), attn_out_buf,
                kv_page_indices, kv_page_indptr, kv_last_page_lens,
                attn_ws, stream, layer_window_left,
                /*logits_soft_cap=*/0.f, sm_scale_override);
        } else if (custom_mask_d) {
            ops::launch_attention_flashinfer_prefill_custom_bf16(
                attn_q, cache.k(L), cache.v(L), attn_out_buf,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                custom_mask_d, custom_mask_indptr_d,
                qo_indptr_h, kv_page_indptr_h,
                N, R, cfg.num_attention_heads, cfg.num_key_value_heads, dk,
                cache.page_size(), attn_ws, stream);
        } else {
            ops::launch_attention_flashinfer_prefill_bf16(
                attn_q, cache.k(L), cache.v(L), attn_out_buf,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, cfg.num_attention_heads, cfg.num_key_value_heads, dk,
                cache.page_size(), attn_ws, stream, layer_window_left,
                /*logits_soft_cap=*/0.f, sm_scale_override);
        }

        // Strip the trailing pad cols off the attention output before
        // it feeds the o_proj GEMM (which expects `[N, num_q*head_dim]`,
        // not `[N, num_q*head_dim_kernel]`).
        if (head_dim_padded) {
            kernels::launch_strip_head_dim_bf16(
                attn_out_buf, ws.attn_out.data(),
                N, cfg.num_attention_heads, d, dk, stream);
        }

        if (!post_norm) {
            // Pre-norm: o_proj accumulates into y (residual add).
            ops::gemm_act_x_wt_bf16(cublas.handle(),
                ws.attn_out.data(), layer.o_proj->data(), ws.y.data(),
                N, H, Hq, /*beta=*/1.f);
        } else {
            // Post-norm: o_proj writes to norm_x (a scratch we own here),
            // norm_attn(norm_x) → norm_y, then y += norm_y.
            ops::gemm_act_x_wt_bf16(cublas.handle(),
                ws.attn_out.data(), layer.o_proj->data(), ws.norm_x.data(),
                N, H, Hq, /*beta=*/0.f);
            kernels::launch_rmsnorm_bf16(
                ws.norm_x.data(), layer.attn_norm->data(), ws.norm_y.data(),
                N, H, eps, stream);
            kernels::launch_residual_add_bf16(
                ws.y.data(), ws.norm_y.data(), N * H, stream);
        }

        // MLP block.
        const void* mlp_in = ws.y.data();
        if (!post_norm) {
            kernels::launch_rmsnorm_bf16(
                ws.y.data(), layer.mlp_norm->data(), ws.norm_y.data(),
                N, H, eps, stream);
            mlp_in = ws.norm_y.data();
        }

        ops::gemm_act_x_wt_bf16(cublas.handle(),
            mlp_in, layer.gate_proj->data(), ws.gate.data(), N, I, H);
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            mlp_in, layer.up_proj->data(),   ws.up.data(),   N, I, H);
        kernels::launch_swiglu_bf16(
            ws.gate.data(), ws.up.data(), ws.gate.data(),
            N * I, stream);

        if (!post_norm) {
            ops::gemm_act_x_wt_bf16(cublas.handle(),
                ws.gate.data(), layer.down_proj->data(), ws.y.data(),
                N, H, I, /*beta=*/1.f);
        } else {
            // Post-norm MLP: down_proj → norm_x scratch, norm_mlp, += y.
            ops::gemm_act_x_wt_bf16(cublas.handle(),
                ws.gate.data(), layer.down_proj->data(), ws.norm_x.data(),
                N, H, I, /*beta=*/0.f);
            kernels::launch_rmsnorm_bf16(
                ws.norm_x.data(), layer.mlp_norm->data(), ws.norm_y.data(),
                N, H, eps, stream);
            kernels::launch_residual_add_bf16(
                ws.y.data(), ws.norm_y.data(), N * H, stream);
        }
    }

    kernels::launch_rmsnorm_bf16(
        ws.y.data(), w.final_norm->data(), ws.norm_x.data(),
        N, H, eps, stream);
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), w.lm_head->data(), ws.logits.data(),
        N, V, H);
}

}  // namespace pie_cuda_driver::model
