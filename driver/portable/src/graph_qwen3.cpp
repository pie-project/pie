#include "graph_qwen3.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "arch_spec.hpp"
#include "plan.hpp"

namespace pie_portable_driver {

namespace {

// Gemma 3n KV-share source. Mirrors graph_gemma4.cpp's gemma4_kv_source
// helper (and the HF transformers logic): a "shared" layer reads K/V
// from the most recent earlier non-shared layer with the same attention
// type. `pattern[il] == 's'` for sliding, `'g'` for full.
inline std::int32_t gemma3n_kv_source(const std::string& pattern,
                                       std::int32_t il,
                                       std::int32_t first_shared) {
    if (il < first_shared) return il;
    const char target = pattern[il];
    for (std::int32_t i = first_shared - 1; i >= 0; --i) {
        if (pattern[i] == target) return i;
    }
    return il;  // unreachable on well-formed configs
}

// Gemma 3n AltUp: compute the per-token modality vector that drives the
// predict / correct coefficient matrices for one stream. Mirrors
// `Gemma3nTextAltUp.compute_router_modalities`:
//   tanh( modality_router @ ( router_norm(x) * (1 / hidden_size) ) )
// Returns a [num_altup_inputs, n_total] tensor.
ggml_tensor* compute_altup_modalities(ggml_context* ctx,
                                       ggml_tensor* x,
                                       const LayerWeights& L,
                                       const ArchSpec& spec,
                                       const Hparams& h) {
    auto* normed = ggml_rms_norm(ctx, x, h.rms_norm_eps);
    normed = norm_scale(ctx, normed, L.altup_router_norm,
                        spec.norm_weight_plus_one);
    normed = ggml_scale(ctx, normed,
                        1.0f / static_cast<float>(h.hidden_size));
    auto* routed = ggml_mul_mat(ctx, L.altup_router, normed);  // [4, n_total]
    return ggml_tanh(ctx, routed);
}

}  // namespace

GraphResult build_qwen3_graph(ggml_context* ctx,
                              const Model& model,
                              KvCachePaged& kv,
                              const Executor::BatchPlan& plan) {
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

    GraphInputs in = declare_graph_inputs(ctx, plan, n_total, n_req,
                                          model.supports_paged_attn_ext());

    // Qwen3-VL M-RoPE: widen pos_input to 4× so ggml_rope_multi reads 4 position
    // axes per token (t,h,w + unused 4th). graph_common fills it global
    // axis-major from plan.mrope_positions_i32 (image/video tokens carry [t,h,w];
    // text tokens [p,p,p], which reduces mrope to plain RoPE). See graph_qwen3_5.
    if (h.use_mrope) {
        in.pos_input = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_total * 4);
        ggml_set_name(in.pos_input, "inp_pos_mrope");
        ggml_set_input(in.pos_input);
    }

    // ---- Embed ---------------------------------------------------------------
    auto* embd = ggml_get_rows(ctx, w.tok_embd, in.tok_input);

    // Multimodal (Qwen3-VL): run the vision encoder over the image patches and
    // splice its soft-token embeddings into the token-embedding rows. Gated so
    // text-only models / batches never build the vision subgraph. The deepstack
    // merger outputs are kept for per-layer injection in the decoder loop.
    std::vector<ggml_tensor*> deepstack_inject;  // [k] added after decoder layer k
    if (plan.has_images && w.vision.present && in.vis_pixels) {
        VisionEncodeResult vres = build_qwen3vl_vision_graph(
            ctx, w.vision, h, in.vis_pixels, in.vis_pos_embed,
            in.vis_rope_cos, in.vis_rope_sin, in.vis_attn_mask, plan.vis_n_patch);
        // embd row width is `hidden`; the merger output is out_hidden (== hidden
        // for Qwen3-VL). Cast f32 encoder output to the embedding dtype, then
        // overwrite the placeholder rows at vis_img_rows.
        ggml_tensor* emb_rows = ggml_cast(ctx, vres.embeddings, embd->type);
        embd = ggml_set_rows(ctx, embd, emb_rows, in.vis_img_rows);
        // Keep the raw (f32) deepstack outputs; cast to the residual-stream
        // dtype at injection time (it differs from the embedding dtype).
        deepstack_inject.assign(vres.deepstack.begin(), vres.deepstack.end());
    }

    auto* inpL = embd;

    // Gemma family: multiply the embedding by sqrt(hidden_size) before
    // entering the layers (matches HF transformers' Gemma forward).
    if (spec.scale_embed_by_sqrt_d) {
        const float embed_scale = std::sqrt(static_cast<float>(h.hidden_size));
        inpL = ggml_scale(ctx, inpL, embed_scale);
    }

    // Default Q scaling = 1/sqrt(head_dim). Gemma2/3 override with
    // 1/sqrt(query_pre_attn_scalar). Gemma3n absorbs the dot-product
    // scale into q_norm and uses unit attention scale (1.0).
    // 0 = use head_dim default.
    const float kq_scale =
        spec.gemma4_unit_sm_scale
            ? 1.0f
            : (spec.query_pre_attn_scalar > 0.0f
                ? 1.0f / std::sqrt(spec.query_pre_attn_scalar)
                : 1.0f / std::sqrt(static_cast<float>(head_dim)));

    // ── Gemma 3n: AltUp init ──
    // The model maintains 4 parallel hidden streams. Stream `altup_active_idx`
    // (=0) is the embedding itself; the other 3 are init'd by projecting the
    // active embedding through `altup_projections.{0..2}`. HF additionally
    // applies a magnitude correction (target_magnitude / sqrt(mean(x^2)))
    // after each projection; we skip that here for v1 (the AltUp coef clip
    // bounds the per-step drift and keeps activations stable enough for
    // first-token correctness; full magnitude correction is a follow-up).
    const bool use_altup =
        (h.arch == PieArch::Gemma3n)
        && h.altup_num_inputs > 1
        && w.altup_proj_0 && w.altup_proj_1 && w.altup_proj_2;
    const std::int32_t altup_n      = use_altup ? h.altup_num_inputs : 1;
    const std::int32_t altup_active = use_altup ? h.altup_active_idx : 0;
    std::vector<ggml_tensor*> streams(altup_n, nullptr);
    if (use_altup) {
        // active stream stays at index `altup_active`; the rest are projections.
        streams[altup_active] = inpL;
        ggml_tensor* proj_w[3] = {w.altup_proj_0, w.altup_proj_1, w.altup_proj_2};

        // Per HF: each projected stream is rescaled to match the active
        // stream's per-token RMS magnitude. Without this, stream[1..3] can
        // sit far off the active stream's scale and the per-layer predict /
        // correct dynamics never recover (output collapses to noise).
        auto* active_sq      = ggml_sqr(ctx, inpL);
        auto* active_mean_sq = ggml_mean(ctx, active_sq);              // [1, n_total]
        auto* target_mag     = ggml_sqrt(ctx, active_mean_sq);

        std::int32_t pi = 0;
        for (std::int32_t i = 0; i < altup_n; ++i) {
            if (i == altup_active) continue;
            auto* projected = ggml_mul_mat(ctx, proj_w[pi++], inpL);
            auto* p_sq      = ggml_sqr(ctx, projected);
            auto* p_mean_sq = ggml_mean(ctx, p_sq);
            // sqrt(max(mean, 1e-5)) ≈ sqrt(mean + 1e-5) for our magnitudes.
            auto* p_mag = ggml_sqrt(ctx,
                ggml_scale_bias(ctx, p_mean_sq, 1.0f, 1e-5f));
            // scale = target_mag / p_mag, broadcast over hidden
            auto* scale = ggml_div(ctx, target_mag, p_mag);
            streams[i] = ggml_mul(ctx, projected, scale);
        }
    } else {
        streams[0] = inpL;
    }

    // ── Gemma 3n: KV-share state ──
    // Last `num_kv_shared_layers` layers reuse upstream non-shared layer's
    // K/V (matched by attention type, see `gemma3n_kv_source`). Track the
    // post-`set_rows` k/v_cached tensors per non-shared layer so shared
    // layers can pick them up by index. Non-Gemma3n archs leave this
    // unused.
    const bool use_kv_share =
        (h.arch == PieArch::Gemma3n)
        && spec.gemma4_first_shared > 0
        && spec.gemma4_first_shared < h.num_hidden_layers
        && static_cast<std::int32_t>(spec.layer_pattern.size())
               == h.num_hidden_layers;
    std::vector<ggml_tensor*> live_k(h.num_hidden_layers, nullptr);
    std::vector<ggml_tensor*> live_v(h.num_hidden_layers, nullptr);

    // ── Gemma 3n: Per-Layer Embeddings (PLE) setup ──
    // The model maintains an auxiliary token-embedding table indexed at the
    // input ids (`embed_tokens_per_layer`, shape [vocab_per_layer,
    // num_layers * ple_dim]) plus a context projection
    // (`per_layer_model_projection`, hidden → num_layers*ple_dim) of the
    // active embedding. Their per-layer slices feed the per-layer PLE
    // residual that the inactive AltUp streams pick up after each block.
    // Mirrors `Gemma3nTextModel.project_per_layer_inputs`.
    ggml_tensor* per_layer_inputs = nullptr;
    if (use_altup && h.gemma4_ple_dim > 0
        && w.ple_token_embed && w.ple_model_proj && w.ple_model_norm) {
        const std::int32_t ple_dim   = h.gemma4_ple_dim;
        const std::int32_t n_layers  = h.num_hidden_layers;
        const float ple_token_norm = std::sqrt(static_cast<float>(ple_dim));
        const float ple_proj_norm  = 1.0f /
            std::sqrt(static_cast<float>(h.hidden_size));
        const float ple_combine    = 1.0f / std::sqrt(2.0f);

        // Token-identity component, scaled by sqrt(ple_dim).
        auto* tok_emb = ggml_get_rows(ctx, w.ple_token_embed, in.tok_input);
        tok_emb = ggml_reshape_3d(ctx, tok_emb, ple_dim, n_layers, n_total);
        tok_emb = ggml_scale(ctx, tok_emb, ple_token_norm);

        // Context component: per_layer_model_projection @ active_embedding,
        // scaled by 1/sqrt(hidden_size), reshaped per-layer, then
        // RMSNormed (per-layer-projection-norm, weight stored centered at
        // 1 → plus_one path matches gemma family convention).
        auto* ctx_proj = ggml_mul_mat(ctx, w.ple_model_proj,
                                      streams[altup_active]);
        ctx_proj = ggml_reshape_3d(ctx, ctx_proj, ple_dim, n_layers, n_total);
        ctx_proj = ggml_scale(ctx, ctx_proj, ple_proj_norm);
        ctx_proj = ggml_rms_norm(ctx, ctx_proj, h.rms_norm_eps);
        ctx_proj = norm_scale(ctx, ctx_proj, w.ple_model_norm,
                              spec.norm_weight_plus_one);

        per_layer_inputs = ggml_add(ctx, ctx_proj, tok_emb);
        per_layer_inputs = ggml_scale(ctx, per_layer_inputs, ple_combine);
    }

    for (std::int32_t il = 0; il < h.num_hidden_layers; ++il) {
        const auto& L = w.layers[il];

        // Gemma 3n KV-share: shared layers reuse upstream non-shared
        // layer's K/V (matched by attention type via layer_pattern).
        const bool is_shared = use_kv_share && il >= spec.gemma4_first_shared;
        const std::int32_t kv_layer = is_shared
            ? gemma3n_kv_source(spec.layer_pattern, il, spec.gemma4_first_shared)
            : il;

        // ── Gemma 3n: AltUp predict ──
        // For each layer, derive 4 "predicted" streams from the current
        // streams using a per-token 4×4 linear combination. The matrix is
        // computed from a small modality vector routed off the active
        // stream's content. Block runs on `predictions[active_idx]`; the
        // correction step closes the loop after the block.
        std::vector<ggml_tensor*> predictions(altup_n, nullptr);
        if (use_altup
            && L.altup_predict_coefs && L.altup_router && L.altup_router_norm) {
            auto* modalities = compute_altup_modalities(
                ctx, streams[altup_active], L, spec, h);
            // [4, n_total] @ [4, 16] (linear weight stored [in=4, out=16])
            // → [16, n_total]; reshape to [4, 4, n_total]. The reshape's
            // inner-most 4 (ne[0]) iterates the input-stream index `q`,
            // ne[1] the output-stream index `p` — matching HF's
            // permute(0,1,3,2) on a [..., 4, 4] reshape of the linear's
            // [..., 16] output (so a per-token entry coef[p, q, t] lives
            // at offset p*nb1 + q*nb0 + t*nb2 in the contiguous result).
            auto* coefs_flat = ggml_mul_mat(
                ctx, L.altup_predict_coefs, modalities);  // [16, n_total]
            auto* coefs_3d = ggml_reshape_3d(
                ctx, coefs_flat,
                /*ne0=*/altup_n, /*ne1=*/altup_n, /*ne2=*/n_total);
            for (std::int32_t p = 0; p < altup_n; ++p) {
                ggml_tensor* acc = streams[p];  // residual: predictions[p] += streams[p]
                for (std::int32_t q = 0; q < altup_n; ++q) {
                    auto* c_pq = ggml_view_2d(
                        ctx, coefs_3d,
                        /*ne0=*/1, /*ne1=*/n_total,
                        /*nb1=*/coefs_3d->nb[2],
                        /*offset=*/p * coefs_3d->nb[1] + q * coefs_3d->nb[0]);
                    auto* term = ggml_mul(ctx, streams[q], c_pq);
                    acc = ggml_add(ctx, acc, term);
                }
                predictions[p] = acc;
            }
            inpL = predictions[altup_active];
        }

        auto* inpSA = inpL;

        // Pre-attention norm. Absent on post-norm-only archs (olmo3),
        // where L.attn_norm is null and we feed the residual stream
        // straight into Q/K/V projections.
        auto* cur = inpL;
        ggml_tensor* active_normed = nullptr;  // Gemma 3n: cached for Laurel
        if (L.attn_norm) {
            cur = ggml_rms_norm(ctx, cur, h.rms_norm_eps);
            cur = norm_scale(ctx, cur, L.attn_norm, spec.norm_weight_plus_one);
            active_normed = cur;
        }

        auto* Q = ggml_mul_mat(ctx, L.q_proj, cur);
        // Shared Gemma 3n layers skip K/V projection — they reuse upstream
        // layer's KV cache by index. Q is still computed on every layer.
        ggml_tensor* K = is_shared ? nullptr : ggml_mul_mat(ctx, L.k_proj, cur);
        ggml_tensor* V = is_shared ? nullptr : ggml_mul_mat(ctx, L.v_proj, cur);

        // M9: optional LoRA delta for q/k/v projections (o_proj is below).
        if (plan.active_adapter
            && static_cast<std::size_t>(il) < plan.active_adapter->layers().size()) {
            const auto& AL = plan.active_adapter->layers()[il];
            const float s = plan.active_adapter->scale();
            Q = apply_lora_delta(ctx, Q, AL.q_a, AL.q_b, cur, s);
            if (!is_shared) {
                K = apply_lora_delta(ctx, K, AL.k_a, AL.k_b, cur, s);
                V = apply_lora_delta(ctx, V, AL.v_a, AL.v_b, cur, s);
            }
        }

        // Optional QKV bias (qwen2). 1D bias vector broadcasts along ne[1]
        // (the n_total token dim) — same as flashinfer / HF.
        if (spec.has_qkv_bias) {
            if (L.q_proj_b) Q = add_with_cast(ctx, Q, L.q_proj_b);
            if (!is_shared) {
                if (L.k_proj_b) K = add_with_cast(ctx, K, L.k_proj_b);
                if (L.v_proj_b) V = add_with_cast(ctx, V, L.v_proj_b);
            }
        }

        // Olmo3 normalizes the flat Q/K vectors (one global RMS over
        // hidden_size) BEFORE the per-head reshape. Weight shapes are
        // [hidden_size] / [kv_dim].
        if (spec.has_qk_norm && spec.qk_norm_full) {
            Q = ggml_rms_norm(ctx, Q, h.rms_norm_eps);
            Q = norm_scale(ctx, Q, L.q_norm, spec.norm_weight_plus_one);
            if (!is_shared) {
                K = ggml_rms_norm(ctx, K, h.rms_norm_eps);
                K = norm_scale(ctx, K, L.k_norm, spec.norm_weight_plus_one);
            }
        }

        Q = ggml_reshape_3d(ctx, Q, head_dim, n_q_heads,  n_total);
        if (!is_shared) {
            K = ggml_reshape_3d(ctx, K, head_dim, n_kv_heads, n_total);
            V = ggml_reshape_3d(ctx, V, head_dim, n_kv_heads, n_total);
        }

        // Per-head Q/K-norm (qwen3, gemma3). Weight is [head_dim] and
        // broadcasts over heads/tokens after the reshape above.
        if (spec.has_qk_norm && !spec.qk_norm_full) {
            Q = ggml_rms_norm(ctx, Q, h.rms_norm_eps);
            Q = norm_scale(ctx, Q, L.q_norm, spec.norm_weight_plus_one);
            if (!is_shared) {
                K = ggml_rms_norm(ctx, K, h.rms_norm_eps);
                K = norm_scale(ctx, K, L.k_norm, spec.norm_weight_plus_one);
            }
        }

        // Gemma 3n: V-norm. Per HF Gemma3nTextAttention.v_norm with
        // `with_scale=False`: pure RMS-norm (no learnable weight) on V
        // BEFORE the KV-cache write, applied per-head over head_dim. Without
        // this the cached V channels carry the un-normalized magnitudes the
        // o_proj weights weren't trained against; attention output ends up
        // off-distribution and the model degrades to fluent-but-incoherent
        // text. (Mirrors graph_gemma4's v_norm path; gemma4_v_norm flag
        // reused since the wiring is identical.)
        if (spec.gemma4_v_norm && !is_shared) {
            V = ggml_rms_norm(ctx, V, h.rms_norm_eps);
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
        if (h.use_mrope) {
            // Qwen3-VL M-RoPE: sectioned multi-axis rope over the 4×-wide
            // pos_input. Sections [t,h,w,0] sum to head_dim/2; IMROPE when the
            // config sets mrope_interleaved. For text rows (t=h=w=p) this is
            // identical to the plain-RoPE branch below.
            int mrope_sections[4] = {
                h.qwen35_mrope_section[0], h.qwen35_mrope_section[1],
                h.qwen35_mrope_section[2], 0};
            const int mrope_mode = h.qwen35_mrope_interleaved
                ? GGML_ROPE_TYPE_IMROPE : GGML_ROPE_TYPE_MROPE;
            Q = ggml_rope_multi(ctx, Q, in.pos_input, c_rope,
                                head_dim, mrope_sections, mrope_mode,
                                rope_n_ctx_orig, layer_rope_theta, rope_freq_scale,
                                rope_ext_factor, rope_attn_factor,
                                rope_beta_fast, rope_beta_slow);
            if (!is_shared) {
                K = ggml_rope_multi(ctx, K, in.pos_input, c_rope,
                                    head_dim, mrope_sections, mrope_mode,
                                    rope_n_ctx_orig, layer_rope_theta, rope_freq_scale,
                                    rope_ext_factor, rope_attn_factor,
                                    rope_beta_fast, rope_beta_slow);
            }
        } else {
            Q = ggml_rope_ext(ctx, Q, in.pos_input, c_rope,
                              head_dim, GGML_ROPE_TYPE_NEOX, rope_n_ctx_orig,
                              layer_rope_theta, rope_freq_scale,
                              rope_ext_factor, rope_attn_factor,
                              rope_beta_fast, rope_beta_slow);
            if (!is_shared) {
                K = ggml_rope_ext(ctx, K, in.pos_input, c_rope,
                                  head_dim, GGML_ROPE_TYPE_NEOX, rope_n_ctx_orig,
                                  layer_rope_theta, rope_freq_scale,
                                  rope_ext_factor, rope_attn_factor,
                                  rope_beta_fast, rope_beta_slow);
            }
        }

        // ---- KV pool write (set_rows scatters by physical row index) -------
        // Shared layers skip the write — they read from upstream's
        // already-written cache slot via `kv_layer`.
        ggml_tensor* k_cached;
        ggml_tensor* v_cached;
        if (!is_shared) {
            // K is contiguous after rope_ext (returns a freshly allocated
            // tensor); V is contiguous after the v_proj mul_mat (and the
            // optional v_norm preserves layout). Skip the unconditional
            // ggml_cont — it was emitting one CPY op per layer per K/V
            // (~48 extra nodes / 24-layer forward), each becoming an extra
            // CUDA kernel launch under sched even though the data is
            // already in the right shape. Fall back to ggml_cont only
            // when actually non-contiguous (defensive — should never
            // trigger on the standard Qwen/Llama path).
            ggml_tensor* k_for_kv = ggml_is_contiguous(K) ? K : ggml_cont(ctx, K);
            ggml_tensor* v_for_kv = ggml_is_contiguous(V) ? V : ggml_cont(ctx, V);
            auto* k_2d = ggml_reshape_2d(ctx, k_for_kv, n_embd_gqa, n_total);
            auto* v_2d = ggml_reshape_2d(ctx, v_for_kv, n_embd_gqa, n_total);
            k_2d = kv.qdq_for_append(ctx, il, k_2d);
            v_2d = kv.qdq_for_append(ctx, il, v_2d);
            k_cached = ggml_set_rows(ctx, kv.k(il), k_2d, in.kv_idxs);
            v_cached = ggml_set_rows(ctx, kv.v(il), v_2d, in.kv_idxs);
            live_k[il] = k_cached;
            live_v[il] = v_cached;
        } else {
            // Read upstream non-shared layer's post-set_rows tensor; this
            // node carries the dependency on that layer's KV write so the
            // graph orders correctly.
            k_cached = live_k[kv_layer];
            v_cached = live_v[kv_layer];
        }

        // ---- Attention -----------------------------------------------------
        ggml_tensor* attn_2d = nullptr;

        if (plan.pure_decode) {
            // Q is [head_dim, n_q_heads, n_total] with n_total == n_req.
            // Reshape (no data move; same memory layout) to
            // [head_dim, 1, n_q_heads, n_request] for ne3 broadcast.
            auto* Q_4d = ggml_reshape_4d(ctx, Q, head_dim, 1, n_q_heads, n_req);

            // Paged-attn fast path. `page_indices` is non-null iff the
            // backend advertised supports_op for ggml_paged_attn_ext at
            // model-load time. attn_sinks is gpt-oss-only; v1 paged
            // kernel doesn't implement sinks, so gpt-oss layers fall
            // through to the materialize path below.
            if (in.page_indices != nullptr && L.attn_sinks == nullptr) {
                // FlashInfer's BatchDecodeWithPagedKVCache wants Q in
                // BF16 (matches K/V dtype family); cast in-graph.
                auto* Q_bf16 = (Q_4d->type == GGML_TYPE_BF16)
                    ? Q_4d : ggml_cast(ctx, Q_4d, GGML_TYPE_BF16);
                auto* attn = ggml_paged_attn_ext(
                    ctx, Q_bf16, k_cached, v_cached,
                    in.page_indices, in.page_indptr, in.last_page_lens,
                    kv.page_size(), head_dim, n_kv_heads,
                    /*sliding_window=*/-1,
                    kq_scale, spec.attn_softcap);
                // Output is BF16 per the constructor (follows q->type).
                // Cast back to F32 for the existing o_proj path.
                auto* attn_f32 = ggml_cast(ctx, attn, GGML_TYPE_F32);
                attn_2d = ggml_reshape_2d(
                    ctx, attn_f32, head_dim * n_q_heads, n_req);
            } else {
                // Materialize: gather all requests' K/V in one call.
                // Result is [n_embd_gqa, max_n_kv * n_req] F32.
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
                // attn shape per ggml.h: [head_dim, n_q_heads, n_batch=1, n_req].
                // flash_attn_ext returns a freshly allocated contiguous tensor;
                // skip the redundant ggml_cont (saves one CPY per layer = 24
                // extra graph nodes on Qwen2.5).
                ggml_tensor* attn_for_reshape =
                    ggml_is_contiguous(attn) ? attn : ggml_cont(ctx, attn);
                attn_2d = ggml_reshape_2d(ctx, attn_for_reshape,
                                          head_dim * n_q_heads, n_req);
            }
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

        // Gemma 3n: Laurel low-rank residual through the MLP block.
        // Per HF Gemma3nTextLaurelBlock + Gemma3nDecoderLayer: laurel takes
        // the input-layernormed active stream `active_normed` (NOT the raw
        // residual), runs a thin [hidden → laurel_rank → hidden] sandwich
        // with a post_norm on the result, sums with itself
        // (`active_normed + post_norm(LR @ LL @ active_normed)`), and the
        // layer averages this with the standard attention residual
        // (`active_prediction + post_attn_norm(attn)`) via 1/√2 scaling.
        if (h.arch == PieArch::Gemma3n && L.laurel_left && L.laurel_right
            && active_normed != nullptr) {
            auto* lx = ggml_mul_mat(ctx, L.laurel_left,  active_normed);
            lx       = ggml_mul_mat(ctx, L.laurel_right, lx);
            if (L.laurel_norm) {
                lx = ggml_rms_norm(ctx, lx, h.rms_norm_eps);
                lx = norm_scale(ctx, lx, L.laurel_norm, spec.norm_weight_plus_one);
            }
            auto* laurel_output = ggml_add(ctx, active_normed, lx);
            ffn_in = ggml_scale(
                ctx,
                ggml_add(ctx, ffn_in, laurel_output),
                1.0f / std::sqrt(2.0f));
        }

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

            // ── Gemma 3n: Gaussian top-k pre-activation sparsity ──
            // Layers in the front of the stack zero out everything below
            // `mean(gate) + std_multiplier * std(gate)` per token, then
            // ReLU. With sparsity=0.95, std_multiplier ≈ icdf(0.95) ≈
            // 1.6448536; only the top ~5% of gate channels survive into
            // the activation. Required for layers 0-9 of E2B/E4B; absent
            // (or sparsity=0) on later layers.
            if (h.arch == PieArch::Gemma3n
                && static_cast<std::size_t>(il) < h.activation_sparsity_pattern.size()
                && h.activation_sparsity_pattern[il] > 0.0f) {
                const float p = h.activation_sparsity_pattern[il];
                // erfcinv-free inverse-CDF for the standard normal:
                // icdf(p) = sqrt(2) * erfinv(2p - 1). std::erf is round-
                // tripped via Newton-style series in libstdc++, but for our
                // small set of fixed p we just compile-time approximate.
                // Using std::sqrt(2) * erfinv(2p-1) where erfinv(x) is
                // computed via a rational approximation (Acklam) — only
                // called once per build_qwen3_graph call so cost is trivial.
                auto icdf_normal = [](float p_) {
                    // 2*p - 1, then erfinv. Acklam's approximation for
                    // 0 < p < 1, accuracy ~1.15e-9 in the central region.
                    const double pp = static_cast<double>(p_);
                    const double a[6] = { -3.969683028665376e+01,
                                           2.209460984245205e+02,
                                          -2.759285104469687e+02,
                                           1.383577518672690e+02,
                                          -3.066479806614716e+01,
                                           2.506628277459239e+00 };
                    const double b[5] = { -5.447609879822406e+01,
                                           1.615858368580409e+02,
                                          -1.556989798598866e+02,
                                           6.680131188771972e+01,
                                          -1.328068155288572e+01 };
                    const double c[6] = { -7.784894002430293e-03,
                                          -3.223964580411365e-01,
                                          -2.400758277161838e+00,
                                          -2.549732539343734e+00,
                                           4.374664141464968e+00,
                                           2.938163982698783e+00 };
                    const double d[4] = {  7.784695709041462e-03,
                                           3.224671290700398e-01,
                                           2.445134137142996e+00,
                                           3.754408661907416e+00 };
                    const double plow  = 0.02425;
                    const double phigh = 1.0 - plow;
                    double q, r;
                    if (pp < plow) {
                        q = std::sqrt(-2.0 * std::log(pp));
                        return static_cast<float>(
                            (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
                            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1));
                    }
                    if (pp <= phigh) {
                        q = pp - 0.5;
                        r = q * q;
                        return static_cast<float>(
                            (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q /
                            (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1));
                    }
                    q = std::sqrt(-2.0 * std::log(1.0 - pp));
                    return static_cast<float>(
                        -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
                        ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1));
                };
                const float std_mult = icdf_normal(p);

                // mean + std (population, unbiased=False).
                auto* g_mean = ggml_mean(ctx, gate);                 // [1, n_total]
                auto* dev    = ggml_sub(ctx, gate, g_mean);          // [hidden, n_total]
                auto* var    = ggml_mean(ctx, ggml_sqr(ctx, dev));   // [1, n_total]
                auto* g_std  = ggml_sqrt(ctx, var);                  // [1, n_total]
                auto* cutoff = ggml_add(ctx, g_mean,
                                        ggml_scale(ctx, g_std, std_mult));
                gate = ggml_relu(ctx, ggml_sub(ctx, gate, cutoff));
            }

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

        // Qwen3-VL deepstack: after decoder layers 0..K-1, add the matching
        // deepstack merger output to the hidden state on the image rows
        // (HF Qwen3VLTextModel._deepstack_process).
        if (static_cast<std::size_t>(il) < deepstack_inject.size() &&
            in.vis_img_rows_i32) {
            ggml_tensor* g = ggml_get_rows(ctx, inpL, in.vis_img_rows_i32);
            ggml_tensor* d = deepstack_inject[static_cast<std::size_t>(il)];
            if (d->type != g->type) d = ggml_cast(ctx, d, g->type);
            g = ggml_add(ctx, g, d);
            inpL = ggml_set_rows(ctx, inpL, g, in.vis_img_rows);
        }

        // ── Gemma 3n: AltUp correct ──
        // Given the activated active stream (the layer's output before any
        // PLE injection) and the pre-block predictions, project the
        // innovation `(activated - predictions[active])` across the 4
        // streams using `correction_coefs(modalities) + 1`. The "+1" makes
        // stream `active_idx` reproduce the activated value when its
        // correction coef is zero (and contributes pure innovation).
        // Then optionally rescale stream `active_idx` by
        // `correct_output_scale` (per-channel). The corrected streams
        // become the input to the next layer's predict step.
        if (use_altup
            && L.altup_correct_coefs && L.altup_router && L.altup_router_norm) {
            auto* activated = inpL;  // [hidden, n_total]
            auto* modalities2 = compute_altup_modalities(
                ctx, activated, L, spec, h);
            auto* corr = ggml_mul_mat(
                ctx, L.altup_correct_coefs, modalities2);  // [4, n_total]
            // corr[p, t] += 1 (folds into the per-stream coefficient).
            corr = ggml_scale_bias(ctx, corr, 1.0f, 1.0f);

            auto* innovation = ggml_sub(
                ctx, activated, predictions[altup_active]);

            std::vector<ggml_tensor*> corrected(altup_n, nullptr);
            for (std::int32_t p = 0; p < altup_n; ++p) {
                auto* c_p = ggml_view_2d(
                    ctx, corr,
                    /*ne0=*/1, /*ne1=*/n_total,
                    /*nb1=*/corr->nb[1],
                    /*offset=*/p * corr->nb[0]);
                auto* term = ggml_mul(ctx, innovation, c_p);
                corrected[p] = ggml_add(ctx, predictions[p], term);
            }

            // ── PLE per-layer residual feed ──
            // Take a working copy of the active-stream correction, optionally
            // rescale by `correct_output_scale` (per-channel; trained init=0
            // so this acts as a residual gate), then run it through the
            // per-layer input gate × PLE-input multiply × per-layer
            // projection × post-norm. The result is added to all *non-active*
            // streams; the active stream is left untouched (HF: only
            // `corrected_predictions[1:] += first_prediction`).
            ggml_tensor* first_prediction = corrected[altup_active];
            if (h.altup_correct_scale && L.altup_correct_scale) {
                auto* sc = (L.altup_correct_scale->type == GGML_TYPE_F32)
                    ? L.altup_correct_scale
                    : ggml_cast(ctx, L.altup_correct_scale, GGML_TYPE_F32);
                first_prediction = ggml_mul(ctx, first_prediction, sc);
            }
            if (per_layer_inputs && L.gemma3n_ple_gate && L.gemma3n_ple_proj
                && L.gemma3n_ple_norm) {
                const std::int32_t ple_dim = h.gemma4_ple_dim;
                const std::size_t  ple_off =
                    static_cast<std::size_t>(il) * ple_dim *
                    ggml_type_size(per_layer_inputs->type);
                // The view straddles a stride of nb[2] in the source —
                // strided reads land on the wrong tokens unless we
                // materialize first. ggml_cast happens to also tolerate
                // strided inputs but we ggml_cont before cast for
                // belt-and-suspenders; same dance graph_gemma4.cpp does.
                auto* ple_signal = ggml_view_2d(
                    ctx, per_layer_inputs,
                    /*ne0=*/ple_dim, /*ne1=*/n_total,
                    /*nb1=*/per_layer_inputs->nb[2],
                    /*offset=*/ple_off);
                auto* ple_signal_f32 = ggml_cast(
                    ctx, ggml_cont(ctx, ple_signal), GGML_TYPE_F32);

                auto* gated = ggml_mul_mat(
                    ctx, L.gemma3n_ple_gate, first_prediction);  // [ple_dim, n_total]
                // hidden_activation = "gelu_pytorch_tanh" on E2B/E4B.
                gated = ggml_gelu(ctx, gated);
                gated = ggml_mul(ctx, gated, ple_signal_f32);
                auto* ple_out = ggml_mul_mat(ctx, L.gemma3n_ple_proj, gated);
                ple_out = ggml_rms_norm(ctx, ple_out, h.rms_norm_eps);
                ple_out = norm_scale(ctx, ple_out, L.gemma3n_ple_norm,
                                     spec.norm_weight_plus_one);

                for (std::int32_t i = 0; i < altup_n; ++i) {
                    if (i == altup_active) continue;
                    corrected[i] = ggml_add(ctx, corrected[i], ple_out);
                }
            }

            streams = std::move(corrected);
            inpL = streams[altup_active];
        } else if (use_altup) {
            // Layer somehow missing the per-layer altup tensors. Keep the
            // active stream advanced with the block output so we don't drop
            // the residual; non-active streams stay frozen for this layer.
            streams[altup_active] = inpL;
        }
    }

    // ── Gemma 3n: Combine 4 streams via the unembed projections ──
    // Mirrors HF's combine: each non-active stream is projected through
    // `altup_unembed_projections.{0..2}`, rescaled to match the active
    // stream's per-token RMS magnitude (so the streams contribute
    // symmetrically to the mean), then averaged with the active stream
    // into a single hidden state for the final norm + LM head.
    if (use_altup
        && w.altup_unembed_proj_0
        && w.altup_unembed_proj_1
        && w.altup_unembed_proj_2) {
        ggml_tensor* unembed_w[3] = {
            w.altup_unembed_proj_0,
            w.altup_unembed_proj_1,
            w.altup_unembed_proj_2,
        };
        auto* active = streams[altup_active];
        auto* active_mean_sq = ggml_mean(ctx, ggml_sqr(ctx, active));
        auto* target_mag = ggml_sqrt(ctx, active_mean_sq);

        auto* combined = active;
        std::int32_t pi = 0;
        for (std::int32_t i = 0; i < altup_n; ++i) {
            if (i == altup_active) continue;
            auto* projected = ggml_mul_mat(ctx, unembed_w[pi++], streams[i]);
            auto* p_mean_sq = ggml_mean(ctx, ggml_sqr(ctx, projected));
            auto* p_mag = ggml_sqrt(ctx,
                ggml_scale_bias(ctx, p_mean_sq, 1.0f, 1e-5f));
            auto* scale = ggml_div(ctx, target_mag, p_mag);
            projected = ggml_mul(ctx, projected, scale);
            combined = ggml_add(ctx, combined, projected);
        }
        combined = ggml_scale(ctx, combined,
                              1.0f / static_cast<float>(altup_n));
        inpL = combined;
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
