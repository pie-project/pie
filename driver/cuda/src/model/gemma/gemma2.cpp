#include "model/gemma/gemma2.hpp"
#include "model/stage_hooks.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/embed.hpp"
#include "kernels/kv_paged.hpp"
#include "kernels/residual_add.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include "kernels/scalar_mul.hpp"
#include "kernels/softcap.hpp"
#include "kernels/swiglu.hpp"

namespace pie_cuda_driver::model {

namespace {

const DeviceTensor& must(const LoadedModel& e, const std::string& name) {
    if (!e.has(name)) {
        throw std::runtime_error("gemma2: missing weight '" + name + "'");
    }
    return e.get(name);
}

}  // namespace

Gemma2Weights bind_gemma2(const LoadedModel& engine) {
    const auto& cfg = engine.hf_config();
    Gemma2Weights w;
    w.embed      = &must(engine, "model.embed_tokens.weight");
    w.final_norm = &must(engine, "model.norm.weight");
    if (engine.has("lm_head.weight")) {
        w.lm_head = &engine.get("lm_head.weight");
    } else if (cfg.tie_word_embeddings) {
        // Gemma-2 ties lm_head to embed_tokens by default.
        w.lm_head = w.embed;
    } else {
        throw std::runtime_error(
            "gemma2: lm_head missing and tie_word_embeddings=false");
    }

    w.layers.resize(static_cast<std::size_t>(cfg.num_hidden_layers));
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        auto& L = w.layers[i];
        L.attn_norm_pre  = &must(engine, p + "input_layernorm.weight");
        L.attn_norm_post = &must(engine, p + "post_attention_layernorm.weight");
        L.mlp_norm_pre   = &must(engine, p + "pre_feedforward_layernorm.weight");
        L.mlp_norm_post  = &must(engine, p + "post_feedforward_layernorm.weight");
        L.q_proj = &must(engine, p + "self_attn.q_proj.weight");
        L.k_proj = &must(engine, p + "self_attn.k_proj.weight");
        L.v_proj = &must(engine, p + "self_attn.v_proj.weight");
        L.o_proj = &must(engine, p + "self_attn.o_proj.weight");
        L.gate_proj = &must(engine, p + "mlp.gate_proj.weight");
        L.up_proj   = &must(engine, p + "mlp.up_proj.weight");
        L.down_proj = &must(engine, p + "mlp.down_proj.weight");
    }
    return w;
}

Gemma2Weights bind_gemma3(const LoadedModel& engine) {
    // Gemma-3 ships the same weight schema as Gemma-2 plus per-head
    // q_norm / k_norm. We delegate the bulk of binding and just patch
    // the qk-norm fields layer-by-layer.
    Gemma2Weights w = bind_gemma2(engine);
    const auto& cfg = engine.hf_config();
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        auto& L = w.layers[i];
        L.q_norm = &must(engine, p + "self_attn.q_norm.weight");
        L.k_norm = &must(engine, p + "self_attn.k_norm.weight");
    }
    return w;
}

void gemma2_forward_paged(
    const Gemma2Weights& w,
    const HfConfig& cfg,
    const Gemma2ForwardCfg& fwd_cfg,
    Workspace& ws,
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
    // TP-local dims. tp_size == 1 reverts to single-GPU shapes.
    const int T  = (fwd_cfg.tp_size > 0) ? fwd_cfg.tp_size : 1;
    const int H  = cfg.hidden_size;
    const int Hq = (cfg.num_attention_heads * cfg.head_dim) / T;
    const int Hk = (cfg.num_key_value_heads * cfg.head_dim) / T;
    const int I  = cfg.intermediate_size / T;
    const int num_q_heads_local  = cfg.num_attention_heads / T;
    const int num_kv_heads_local = cfg.num_key_value_heads / T;
    const int V  = cfg.vocab_size;
    const int d  = cfg.head_dim;
    const float eps = cfg.rms_norm_eps;
    cudaStream_t stream = nullptr;
    NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;

    const bool use_decode_path = is_pure_decode && !fwd_cfg.force_prefill_path;

    // Step 1: embed and Gemma's √hidden_size scale.
    kernels::launch_embed_bf16(
        token_ids, w.embed->data(), ws.y.data(), N, H, V, stream);
    kernels::launch_scalar_mul_bf16(
        ws.y.data(), std::sqrt(static_cast<float>(H)),
        static_cast<std::size_t>(N) * H, stream);

    ops::DecodePlanCachePtr decode_plan;
    if (use_decode_path) {
        decode_plan = ops::make_decode_plan();
        ops::plan_attention_flashinfer_decode(
            *decode_plan, kv_page_indptr_h, R,
            num_q_heads_local, num_kv_heads_local, d,
            cache.page_size(), attn_ws, stream,
            /*enable_cuda_graph=*/true,
            /*full_attention_variant=*/false,
            cache.hnd_layout());
    }

    // flashinfer's attention applies `q * 1/sqrt(head_dim)` internally.
    // Gemma-2 wants `q * 1/sqrt(query_pre_attn_scalar)`. To compensate,
    // multiply q by `sqrt(head_dim) / sqrt(query_pre_attn_scalar)` after
    // the q_proj. When the two values agree (Gemma-2-2B / 27B), this is
    // 1.0 and the scaling is a no-op.
    const float q_pre_scale =
        std::sqrt(static_cast<float>(d) / fwd_cfg.query_pre_attn_scalar);
    const bool need_q_pre_scale = std::abs(q_pre_scale - 1.f) > 1e-6f;

    for (int L = 0; L < cfg.num_hidden_layers; ++L) {
        const auto& layer = w.layers[L];

        // ── Attention block ──
        // 1a. pre-attn norm.
        kernels::launch_rmsnorm_gemma_bf16(
            ws.y.data(), layer.attn_norm_pre->data(), ws.norm_x.data(),
            N, H, eps, stream);

        // 1b. QKV projection.
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.norm_x.data(), layer.q_proj->data(), ws.q.data(), N, Hq, H);
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.norm_x.data(), layer.k_proj->data(), ws.k.data(), N, Hk, H);
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.norm_x.data(), layer.v_proj->data(), ws.v.data(), N, Hk, H);
        invoke_stage_hook(
            StageHookPoint::OnAttnProj, ws.q.data(),
            static_cast<std::uint32_t>(N),
            static_cast<std::uint32_t>(Hq),
            static_cast<std::uint32_t>(L), stream);

        // Per-head q/k RMSNorm (Gemma-3+) — applied before the query
        // pre-scale and before RoPE, matching HF's
        // `Gemma3Attention.forward`.
        if (fwd_cfg.use_qk_norm && layer.q_norm) {
            kernels::launch_rmsnorm_gemma_bf16(
                ws.q.data(), layer.q_norm->data(), ws.q.data(),
                N * num_q_heads_local, d, eps, stream);
        }
        if (fwd_cfg.use_qk_norm && layer.k_norm) {
            kernels::launch_rmsnorm_gemma_bf16(
                ws.k.data(), layer.k_norm->data(), ws.k.data(),
                N * num_kv_heads_local, d, eps, stream);
        }

        if (need_q_pre_scale) {
            kernels::launch_scalar_mul_bf16(
                ws.q.data(), q_pre_scale,
                static_cast<std::size_t>(N) * Hq, stream);
        }

        // Per-layer rope_theta (Gemma-3 dual-RoPE). When the layer-vector
        // is empty, every layer uses the global theta — Gemma-2's case.
        const float layer_rope_theta =
            (!fwd_cfg.per_layer_rope_theta.empty() &&
             L < static_cast<int>(fwd_cfg.per_layer_rope_theta.size()))
                ? fwd_cfg.per_layer_rope_theta[L]
                : cfg.rope_theta;

        kernels::launch_rope_bf16(
            ws.q.data(), ws.k.data(), positions,
            N, num_q_heads_local, num_kv_heads_local, d,
            layer_rope_theta, stream);
        auto kv_view = cache.layer_view(L);
        kernels::launch_write_kv_to_pages(
            kv_view, ws.k.data(), ws.v.data(),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            N, R, stream);

        const int layer_window_left =
            (!fwd_cfg.per_layer_window_left.empty() &&
             L < static_cast<int>(fwd_cfg.per_layer_window_left.size()))
                ? fwd_cfg.per_layer_window_left[L]
                : -1;

        if (use_decode_path) {
            ops::dispatch_attention_flashinfer_decode(
                *decode_plan,
                ws.q.data(), kv_view, ws.attn_out.data(),
                kv_page_indices, kv_page_indptr, kv_last_page_lens,
                attn_ws, stream, layer_window_left, fwd_cfg.attn_logit_softcap);
        } else if (custom_mask_d) {
            ops::launch_attention_flashinfer_prefill_custom(
                ws.q.data(), kv_view, ws.attn_out.data(),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                custom_mask_d, custom_mask_indptr_d,
                qo_indptr_h, kv_page_indptr_h,
                N, R, num_q_heads_local, attn_ws, stream);
        } else {
            ops::launch_attention_flashinfer_prefill(
                ws.q.data(), kv_view, ws.attn_out.data(),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, num_q_heads_local, attn_ws, stream,
                layer_window_left, fwd_cfg.attn_logit_softcap);
        }
        invoke_stage_hook(
            StageHookPoint::OnAttn, ws.q.data(),
            static_cast<std::uint32_t>(N),
            static_cast<std::uint32_t>(Hq),
            static_cast<std::uint32_t>(L), stream);

        // 1c. o_proj → norm_x scratch (NOT into y), apply post-attn norm,
        //     then residual-add into y. Under TP this is row-parallel; we
        //     all-reduce the partial sum before the post-attn norm sees it.
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.attn_out.data(), layer.o_proj->data(), ws.norm_x.data(),
            N, H, Hq, /*beta=*/0.f);
        if (T > 1) {
            tp->all_reduce_bf16(ws.norm_x.data(),
                static_cast<std::size_t>(N) * H, ncclSum, stream);
        }
        kernels::launch_rmsnorm_gemma_bf16(
            ws.norm_x.data(), layer.attn_norm_post->data(), ws.norm_y.data(),
            N, H, eps, stream);
        kernels::launch_residual_add_bf16(
            ws.y.data(), ws.norm_y.data(),
            static_cast<std::size_t>(N) * H, stream);

        // ── MLP block ──
        // 2a. pre-MLP norm.
        kernels::launch_rmsnorm_gemma_bf16(
            ws.y.data(), layer.mlp_norm_pre->data(), ws.norm_x.data(),
            N, H, eps, stream);

        // 2b. GeGLU(tanh): gate / up GEMMs, GeLU-tanh-glu, down.
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.norm_x.data(), layer.gate_proj->data(), ws.gate.data(), N, I, H);
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.norm_x.data(), layer.up_proj->data(),   ws.up.data(),   N, I, H);
        kernels::launch_geglu_tanh_bf16(
            ws.gate.data(), ws.up.data(), ws.gate.data(),
            N * I, stream);

        // 2c. down → norm_x scratch, apply post-MLP norm, residual-add.
        // Same row-parallel + all-reduce dance as o_proj.
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.gate.data(), layer.down_proj->data(), ws.norm_x.data(),
            N, H, I, /*beta=*/0.f);
        if (T > 1) {
            tp->all_reduce_bf16(ws.norm_x.data(),
                static_cast<std::size_t>(N) * H, ncclSum, stream);
        }
        kernels::launch_rmsnorm_gemma_bf16(
            ws.norm_x.data(), layer.mlp_norm_post->data(), ws.norm_y.data(),
            N, H, eps, stream);
        kernels::launch_residual_add_bf16(
            ws.y.data(), ws.norm_y.data(),
            static_cast<std::size_t>(N) * H, stream);
    }

    kernels::launch_rmsnorm_gemma_bf16(
        ws.y.data(), w.final_norm->data(), ws.norm_x.data(),
        N, H, eps, stream);
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), w.lm_head->data(), ws.logits.data(),
        N, V, H);

    if (fwd_cfg.final_logit_softcap > 0.f) {
        kernels::launch_logit_softcap_bf16(
            ws.logits.data(), fwd_cfg.final_logit_softcap,
            static_cast<std::size_t>(N) * V, stream);
    }
}

}  // namespace pie_cuda_driver::model
