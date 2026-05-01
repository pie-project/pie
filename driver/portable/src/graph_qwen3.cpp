#include "graph_qwen3.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "arch_spec.hpp"
#include "plan.hpp"

namespace pie_portable_driver {

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
    auto* gf = ggml_new_graph_custom(
        ctx, static_cast<int>(GRAPH_MAX_NODES), /*grads=*/false);

    GraphInputs in = declare_graph_inputs(ctx, plan, n_total, n_req);

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

        // Pre-attention norm. Absent on post-norm-only archs (olmo3),
        // where L.attn_norm is null and we feed the residual stream
        // straight into Q/K/V projections.
        auto* cur = inpL;
        if (L.attn_norm) {
            cur = ggml_rms_norm(ctx, cur, h.rms_norm_eps);
            cur = norm_scale(ctx, cur, L.attn_norm, spec.norm_weight_plus_one);
        }

        auto* Q = ggml_mul_mat(ctx, L.q_proj, cur);
        auto* K = ggml_mul_mat(ctx, L.k_proj, cur);
        auto* V = ggml_mul_mat(ctx, L.v_proj, cur);

        // M9: optional LoRA delta for q/k/v projections (o_proj is below).
        if (plan.active_adapter
            && static_cast<std::size_t>(il) < plan.active_adapter->layers().size()) {
            const auto& AL = plan.active_adapter->layers()[il];
            const float s = plan.active_adapter->scale();
            Q = apply_lora_delta(ctx, Q, AL.q_a, AL.q_b, cur, s);
            K = apply_lora_delta(ctx, K, AL.k_a, AL.k_b, cur, s);
            V = apply_lora_delta(ctx, V, AL.v_a, AL.v_b, cur, s);
        }

        // Optional QKV bias (qwen2). 1D bias vector broadcasts along ne[1]
        // (the n_total token dim) — same as flashinfer / HF.
        if (spec.has_qkv_bias) {
            if (L.q_proj_b) Q = add_with_cast(ctx, Q, L.q_proj_b);
            if (L.k_proj_b) K = add_with_cast(ctx, K, L.k_proj_b);
            if (L.v_proj_b) V = add_with_cast(ctx, V, L.v_proj_b);
        }

        // Olmo3 normalizes the flat Q/K vectors (one global RMS over
        // hidden_size) BEFORE the per-head reshape. Weight shapes are
        // [hidden_size] / [kv_dim].
        if (spec.has_qk_norm && spec.qk_norm_full) {
            Q = ggml_rms_norm(ctx, Q, h.rms_norm_eps);
            Q = norm_scale(ctx, Q, L.q_norm, spec.norm_weight_plus_one);
            K = ggml_rms_norm(ctx, K, h.rms_norm_eps);
            K = norm_scale(ctx, K, L.k_norm, spec.norm_weight_plus_one);
        }

        Q = ggml_reshape_3d(ctx, Q, head_dim, n_q_heads,  n_total);
        K = ggml_reshape_3d(ctx, K, head_dim, n_kv_heads, n_total);
        V = ggml_reshape_3d(ctx, V, head_dim, n_kv_heads, n_total);

        // Per-head Q/K-norm (qwen3, gemma3). Weight is [head_dim] and
        // broadcasts over heads/tokens after the reshape above.
        if (spec.has_qk_norm && !spec.qk_norm_full) {
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
        // YaRN (olmo3, Ministral 3, gpt-oss): non-zero ext_factor switches
        // ggml's RoPE into the smooth-ramp interpolation/extrapolation blend.
        // Plain θ-only RoPE keeps ext_factor=0 / attn_factor=1.
        const bool yarn_on = spec.yarn_n_ctx_orig > 0;
        const std::int32_t rope_n_ctx_orig =
            yarn_on ? spec.yarn_n_ctx_orig : 0;
        const float rope_freq_scale =
            yarn_on ? spec.yarn_freq_scale  : 1.0f;
        const float rope_ext_factor =
            yarn_on ? 1.0f                  : 0.0f;
        const float rope_attn_factor =
            yarn_on ? spec.yarn_attn_factor : 1.0f;
        const float rope_beta_fast =
            yarn_on ? spec.yarn_beta_fast   : 32.0f;
        const float rope_beta_slow =
            yarn_on ? spec.yarn_beta_slow   : 1.0f;
        Q = ggml_rope_ext(ctx, Q, in.pos_input, c_rope,
                          head_dim, GGML_ROPE_TYPE_NEOX, rope_n_ctx_orig,
                          layer_rope_theta, rope_freq_scale,
                          rope_ext_factor, rope_attn_factor,
                          rope_beta_fast, rope_beta_slow);
        K = ggml_rope_ext(ctx, K, in.pos_input, c_rope,
                          head_dim, GGML_ROPE_TYPE_NEOX, rope_n_ctx_orig,
                          layer_rope_theta, rope_freq_scale,
                          rope_ext_factor, rope_attn_factor,
                          rope_beta_fast, rope_beta_slow);

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
            // gpt-oss attention sinks (per Q-head learned scalar). Adds an
            // implicit "attend to nothing" slot that absorbs probability
            // mass when no other key is in-window. Null on archs without
            // sinks. Sinks stored F32 [n_q_heads]; broadcasts over batch.
            if (L.attn_sinks) {
                // ggml_flash_attn_ext_add_sinks asserts F32; sinks are
                // stored as BF16 in safetensors. Cast in-graph.
                auto* sinks_f32 = (L.attn_sinks->type == GGML_TYPE_F32)
                    ? L.attn_sinks
                    : ggml_cast(ctx, L.attn_sinks, GGML_TYPE_F32);
                ggml_flash_attn_ext_add_sinks(attn, sinks_f32);
            }
            // attn shape per ggml.h: [head_dim, n_q_heads, n_batch=1, n_req]
            attn_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, attn),
                                      head_dim * n_q_heads, n_req);
        } else {
            // Slow path: one flash_attn_ext per request, then concat.
            std::vector<ggml_tensor*> attn_out_per_req;
            attn_out_per_req.reserve(n_req);
            for (std::int32_t r = 0; r < n_req; ++r) {
                const auto& R = plan.reqs[r];
                attn_out_per_req.push_back(build_request_flash_attn(
                    ctx, Q, k_cached, v_cached,
                    in.gather_idxs[r], in.masks[r],
                    R.qo_start, R.n_tokens, R.n_kv,
                    head_dim, n_kv_heads, n_q_heads,
                    kq_scale, spec.attn_softcap, L.attn_sinks));
            }
            attn_2d = concat_per_request_attn(
                ctx, attn_out_per_req, head_dim, n_q_heads, n_total);
        }

        auto* attn_out = ggml_mul_mat(ctx, L.o_proj, attn_2d);
        // gpt-oss has an o_proj bias.
        if (L.o_proj_b) {
            attn_out = add_with_cast(ctx, attn_out, L.o_proj_b);
        }

        // M9: optional LoRA delta on o_proj.
        if (plan.active_adapter
            && static_cast<std::size_t>(il) < plan.active_adapter->layers().size()) {
            const auto& AL = plan.active_adapter->layers()[il];
            attn_out = apply_lora_delta(ctx, attn_out, AL.o_a, AL.o_b,
                                        attn_2d, plan.active_adapter->scale());
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
        // Olmo3 has no pre-FFN norm (post-norm-only); L.ffn_norm is null and
        // the FFN sees the residual stream directly.
        cur = ffn_in;
        if (L.ffn_norm) {
            cur = ggml_rms_norm(ctx, cur, h.rms_norm_eps);
            cur = norm_scale(ctx, cur, L.ffn_norm, spec.norm_weight_plus_one);
        }

        ggml_tensor* ffn_out;
        if (spec.n_experts > 0) {
            // MoE dispatch (Mixtral / Qwen-MoE / GPT-OSS / DeepSeek-style).
            // gpt-oss: SwigluOai + per-expert biases + router bias.
            // Mixtral / Qwen3-MoE: standard SiLU SwiGLU, no biases.
            const MoeActivation moe_act = (h.arch == PieArch::GptOss)
                ? MoeActivation::SwigluOai
                : (spec.ffn_use_gelu ? MoeActivation::Gelu
                                     : MoeActivation::Silu);
            ffn_out = build_moe_ffn(ctx, cur,
                                    L.moe_router,
                                    L.moe_gate_exps,
                                    L.moe_up_exps,
                                    L.moe_down_exps,
                                    spec.n_experts,
                                    spec.n_experts_per_tok,
                                    moe_act,
                                    spec.moe_norm_topk,
                                    L.moe_router_b,
                                    L.moe_gate_exps_b,
                                    L.moe_up_exps_b,
                                    L.moe_down_exps_b);
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

    ggml_tensor* tokens_out  = nullptr;
    ggml_tensor* top_k_idx   = nullptr;
    ggml_tensor* top_k_probs = nullptr;
    if (plan.all_greedy) {
        // Greedy fast path: argmax along vocab axis on the GPU. Caller
        // downloads only n_slots i32 ids; the F32 logits buffer never
        // needs to be a graph output, so gallocr can avoid the
        // ~vocab*n_slots*4-byte materialization. (CPU argmax in our
        // host sampler uses first-wins-on-tie; the CUDA reduction picks
        // a different tied index in rare exact-tie cases — both are
        // valid greedy choices.)
        tokens_out = ggml_argmax(ctx, logits);
        ggml_set_name(tokens_out, "tokens_out");
        ggml_set_output(tokens_out);
        ggml_build_forward_expand(gf, tokens_out);
    } else if (plan.uniform_top_sample) {
        // Non-greedy uniform fast path: temperature-scale + softmax →
        // probs, take top-K indices, gather K probs. Caller downloads
        // only K * n_slots * 8 bytes (vs vocab * n_slots * 4 bytes for
        // the slow path) and finalizes per-slot top-p / min-p / sample
        // host-side.
        const float inv_t = 1.0f / plan.reqs[0].sampler.temperature;
        ggml_tensor* probs = ggml_soft_max_ext(ctx, logits, /*mask=*/ nullptr,
                                               /*scale=*/ inv_t, /*max_bias=*/ 0.0f);
        // ggml_top_k returns indices [K, n_slots] sorted descending.
        top_k_idx = ggml_top_k(ctx, probs, plan.uniform_top_k);

        // Gather the K probabilities matching those indices, mirroring
        // the get_rows trick used by the MoE router (build_moe_ffn).
        // probs reshaped to [1, vocab, n_slots]; get_rows along ne[1]
        // with index [K, n_slots] yields [1, K, n_slots] → reshape.
        ggml_tensor* probs_3d = ggml_reshape_3d(
            ctx, probs, 1, h.vocab_size, n_req);
        ggml_tensor* gathered = ggml_get_rows(ctx, probs_3d, top_k_idx);
        top_k_probs = ggml_reshape_2d(ctx, gathered, plan.uniform_top_k, n_req);

        ggml_set_name(top_k_idx, "top_k_idx");
        ggml_set_name(top_k_probs, "top_k_probs");
        ggml_set_output(top_k_idx);
        ggml_set_output(top_k_probs);
        ggml_build_forward_expand(gf, top_k_idx);
        ggml_build_forward_expand(gf, top_k_probs);
    } else {
        ggml_set_name(logits, "logits");
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);
    }

    GraphResult res{};
    res.gf = gf;
    // Only one output is materialized depending on the chosen sampling
    // path. The other two stay null and gallocr skips their backing
    // buffers (most importantly the [vocab, n_slots] F32 logits block).
    if (plan.all_greedy) {
        res.tokens_out = tokens_out;
    } else if (plan.uniform_top_sample) {
        res.top_k_idx   = top_k_idx;
        res.top_k_probs = top_k_probs;
    } else {
        res.logits = logits;
    }
    res.in = std::move(in);
    return res;
}

}  // namespace pie_portable_driver
