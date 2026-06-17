#include "graph_common.hpp"

#include <cstring>
#include <string>

#include <ggml-backend.h>

#include "plan.hpp"  // MASK_PAD

// Hook into the FlashInfer-backed paged-attn registration in
// driver/portable/src/paged_attn_cuda.cu. The .cu file is only compiled
// when GGML_CUDA=ON; otherwise we fall back to a no-op (the paged path
// is never taken on CPU/Vulkan/Metal builds anyway, so the call is
// unreachable but the linker still needs a symbol).
extern "C" void pie_paged_attn_set_host_indptr(
    const std::int32_t* indptr, std::size_t n_req_plus_one);

#ifndef PIE_PORTABLE_HAS_CUDA
extern "C" void pie_paged_attn_set_host_indptr(
        const std::int32_t* /*indptr*/, std::size_t /*n_req_plus_one*/) {
    // Stub: paged path can't be taken without a registered kernel, so
    // upload_graph_inputs never reaches this. Defined to satisfy the
    // linker on non-CUDA builds.
}
#endif

namespace pie_portable_driver {

GraphInputs declare_graph_inputs(ggml_context* ctx,
                                 const Executor::BatchPlan& plan,
                                 std::int32_t n_total,
                                 std::int32_t n_req,
                                 bool         supports_paged_attn_ext) {
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

    // Vision (multimodal) side-inputs. Dims derive from the host-computed
    // BatchPlan arrays (hidden/head_dim recovered from their lengths).
    if (plan.has_images && plan.vis_n_patch > 0) {
        const std::int64_t np = plan.vis_n_patch;
        const std::int64_t hidden =
            np > 0 ? static_cast<std::int64_t>(plan.vis_pos_embed.size()) / np : 0;
        const std::int64_t hd =
            np > 0 ? static_cast<std::int64_t>(plan.vis_rope_cos.size()) / np : 0;
        in.vis_pixels = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, plan.vis_patch_dim, np);
        ggml_set_name(in.vis_pixels, "vis_pixels");
        ggml_set_input(in.vis_pixels);
        in.vis_pos_embed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, np);
        ggml_set_name(in.vis_pos_embed, "vis_pos_embed");
        ggml_set_input(in.vis_pos_embed);
        in.vis_rope_cos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, np);
        ggml_set_name(in.vis_rope_cos, "vis_rope_cos");
        ggml_set_input(in.vis_rope_cos);
        in.vis_rope_sin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, np);
        ggml_set_name(in.vis_rope_sin, "vis_rope_sin");
        ggml_set_input(in.vis_rope_sin);
        in.vis_img_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, plan.vis_n_token);
        ggml_set_name(in.vis_img_rows, "vis_img_rows");
        ggml_set_input(in.vis_img_rows);
        // vis_img_rows_i32 is consumed only by Qwen3-VL deepstack get_rows.
        // The Gemma-4 path (pool matrix present) never reads it, and allocating
        // an unused input leaves it bufferless -> upload asserts. So gate it.
        if (plan.vis_pool_matrix.empty()) {
            in.vis_img_rows_i32 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, plan.vis_n_token);
            ggml_set_name(in.vis_img_rows_i32, "vis_img_rows_i32");
            ggml_set_input(in.vis_img_rows_i32);
        }
        if (!plan.vis_pool_matrix.empty()) {  // Gemma-4 group-mean pool
            in.vis_pool_matrix = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, np,
                                                    plan.vis_n_token);
            ggml_set_name(in.vis_pool_matrix, "vis_pool_matrix");
            ggml_set_input(in.vis_pool_matrix);
        }
        if (!plan.vis_attn_mask.empty()) {  // multi-image block-diagonal mask
            in.vis_attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, np, np);
            ggml_set_name(in.vis_attn_mask, "vis_attn_mask");
            ggml_set_input(in.vis_attn_mask);
        }
    }

    // Audio (Gemma-4 Conformer) side-inputs. One encoder per clip; the
    // sinusoidal rel-pos encoding `aud_pe` is shared across clips.
    if (plan.has_audio && !plan.aud_clips.empty()) {
        const std::int64_t P = plan.aud_dwin + 1;  // pe rows (= context_left)
        const std::int64_t hidden =
            P == 0 ? 0 : static_cast<std::int64_t>(plan.aud_pe.size()) / P;
        in.aud_pe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, P);
        ggml_set_name(in.aud_pe, "aud_pe");
        ggml_set_input(in.aud_pe);
        for (std::size_t c = 0; c < plan.aud_clips.size(); ++c) {
            const auto& clip = plan.aud_clips[c];
            const std::string sfx = std::to_string(c);
            ggml_tensor* feat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                   plan.aud_n_mel, clip.n_frame);
            ggml_set_name(feat, ("aud_features" + sfx).c_str());
            ggml_set_input(feat);
            in.aud_features.push_back(feat);
            ggml_tensor* wm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                 plan.aud_dwin, clip.n_token);
            ggml_set_name(wm, ("aud_win_mask" + sfx).c_str());
            ggml_set_input(wm);
            in.aud_win_mask.push_back(wm);
            ggml_tensor* rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, clip.n_token);
            ggml_set_name(rows, ("aud_rows" + sfx).c_str());
            ggml_set_input(rows);
            in.aud_rows.push_back(rows);
        }
    }

    if (plan.pure_decode) {
        // Either paged-attn inputs (page_indices/indptr/last_page_lens)
        // or the materialize path's packed_gather — never both. The
        // allocator skips tensors not reached from graph nodes, so
        // allocating an unused input tensor leaves it bufferless and
        // breaks upload.
        const bool use_paged = supports_paged_attn_ext &&
                               !plan.page_indices_i32.empty();

        if (!use_paged) {
            // Materialize path: single packed gather covers all requests.
            in.packed_gather = ggml_new_tensor_1d(
                ctx, GGML_TYPE_I32,
                static_cast<std::int64_t>(plan.max_n_kv) * n_req);
            ggml_set_name(in.packed_gather, "kv_gather.packed");
            ggml_set_input(in.packed_gather);
        }

        // packed_mask is consumed only by the materialize path; FlashInfer
        // handles SWA natively via window_left so paged path doesn't need it.
        // Allocating it on the paged path would leave it unused and trip
        // the allocator-skip → bufferless upload bug.
        if (!use_paged) {
            in.packed_mask = ggml_new_tensor_4d(
                ctx, GGML_TYPE_F16, plan.max_n_kv,
                static_cast<std::int64_t>(MASK_PAD), 1, n_req);
            ggml_set_name(in.packed_mask, "kq_mask.packed");
            ggml_set_input(in.packed_mask);

            // Optional no-SWA companion (Gemma 4 full-attention layers).
            if (!plan.packed_mask_full_f16.empty()) {
                in.packed_mask_full = ggml_new_tensor_4d(
                    ctx, GGML_TYPE_F16, plan.max_n_kv,
                    static_cast<std::int64_t>(MASK_PAD), 1, n_req);
                ggml_set_name(in.packed_mask_full, "kq_mask.packed.full");
                ggml_set_input(in.packed_mask_full);
            }
        }

        if (use_paged) {
            in.page_indices = ggml_new_tensor_1d(
                ctx, GGML_TYPE_I32,
                static_cast<std::int64_t>(plan.page_indices_i32.size()));
            ggml_set_name(in.page_indices, "kv_page_indices");
            ggml_set_input(in.page_indices);

            in.page_indptr = ggml_new_tensor_1d(
                ctx, GGML_TYPE_I32, n_req + 1);
            ggml_set_name(in.page_indptr, "kv_page_indptr");
            ggml_set_input(in.page_indptr);

            in.last_page_lens = ggml_new_tensor_1d(
                ctx, GGML_TYPE_I32, n_req);
            ggml_set_name(in.last_page_lens, "kv_last_page_lens");
            ggml_set_input(in.last_page_lens);
        }
    } else {
        in.masks.reserve(n_req);
        in.masks_blocksparse.reserve(n_req);
        in.gather_idxs.reserve(n_req);
        for (std::int32_t r = 0; r < n_req; ++r) {
            const auto& R = plan.reqs[r];
            auto* m = ggml_new_tensor_4d(
                ctx, GGML_TYPE_F16, R.n_kv, R.n_tokens_pad, 1, 1);
            ggml_set_name(m, ("kq_mask." + std::to_string(r)).c_str());
            ggml_set_input(m);
            in.masks.push_back(m);

            // Phi-3-small blocksparse companion mask. Allocated only
            // when the request carries one (else null — per-layer code
            // falls back to in.masks[r]).
            if (!R.mask_blocksparse_f16.empty()) {
                auto* mbs = ggml_new_tensor_4d(
                    ctx, GGML_TYPE_F16, R.n_kv, R.n_tokens_pad, 1, 1);
                ggml_set_name(mbs, ("kq_mask_bs." + std::to_string(r)).c_str());
                ggml_set_input(mbs);
                in.masks_blocksparse.push_back(mbs);
            } else {
                in.masks_blocksparse.push_back(nullptr);
            }

            auto* g = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, R.n_kv);
            ggml_set_name(g, ("kv_gather." + std::to_string(r)).c_str());
            ggml_set_input(g);
            in.gather_idxs.push_back(g);
        }
    }
    return in;
}

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

ggml_tensor* build_moe_ffn(ggml_context* ctx,
                           ggml_tensor* cur,
                           ggml_tensor* gate_inp,
                           ggml_tensor* gate_exps,
                           ggml_tensor* up_exps,
                           ggml_tensor* down_exps,
                           std::int32_t n_experts,
                           std::int32_t n_used,
                           MoeActivation act,
                           bool norm_topk,
                           ggml_tensor* gate_inp_b,
                           ggml_tensor* gate_exps_b,
                           ggml_tensor* up_exps_b,
                           ggml_tensor* down_exps_b,
                           float        oai_alpha,
                           float        oai_limit) {
    const std::int64_t n_total = cur->ne[1];

    // ---- 1. Router ------------------------------------------------------
    // logits[e, t] = sum_h gate_inp[h, e] * cur[h, t]   →  [n_experts, n_total]
    ggml_tensor* logits = ggml_mul_mat(ctx, gate_inp, cur);
    if (gate_inp_b) {
        // Cast bias to logits' dtype; mul_mat output is F32, bias may be BF16.
        auto* b = (gate_inp_b->type == logits->type)
            ? gate_inp_b
            : ggml_cast(ctx, gate_inp_b, logits->type);
        logits = ggml_add(ctx, logits, b);
    }
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
    auto cast_to_f32 = [&](ggml_tensor* t) {
        return (t->type == GGML_TYPE_F32) ? t : ggml_cast(ctx, t, GGML_TYPE_F32);
    };
    if (gate_exps_b) {
        gate_out = ggml_add_id(ctx, gate_out, cast_to_f32(gate_exps_b), selected);
    }
    if (up_exps_b) {
        up_out = ggml_add_id(ctx, up_out, cast_to_f32(up_exps_b), selected);
    }

    ggml_tensor* gated = nullptr;
    switch (act) {
        case MoeActivation::Silu:
            gated = ggml_mul(ctx, ggml_silu(ctx, gate_out), up_out);
            break;
        case MoeActivation::Gelu:
            gated = ggml_mul(ctx, ggml_gelu(ctx, gate_out), up_out);
            break;
        case MoeActivation::SwigluOai:
            // gpt-oss: clamp(gate, max=limit), clamp(up, ±limit), then
            // act = silu(α·gate) * (up + β=1). ggml_swiglu_oai handles
            // the whole expression including the clamps.
            gated = ggml_swiglu_oai(ctx, gate_out, up_out, oai_alpha, oai_limit);
            break;
    }

    ggml_tensor* expert_out =
        ggml_mul_mat_id(ctx, down_exps, gated, selected);
    if (down_exps_b) {
        expert_out = ggml_add_id(ctx, expert_out, cast_to_f32(down_exps_b), selected);
    }
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

ggml_tensor* build_request_flash_attn(
    ggml_context* ctx,
    ggml_tensor*  Q,
    ggml_tensor*  k_cached,
    ggml_tensor*  v_cached,
    ggml_tensor*  gather_idx_r,
    ggml_tensor*  mask_r,
    std::int32_t  qo_start,
    std::int32_t  n_tokens,
    std::int32_t  n_kv,
    std::int32_t  head_dim,
    std::int32_t  n_kv_heads,
    std::int32_t  n_q_heads,
    float         kq_scale,
    float         attn_softcap,
    ggml_tensor*  sinks) {
    const std::size_t Q_stride_ne2 =
        static_cast<std::size_t>(head_dim) * n_q_heads *
        ggml_type_size(Q->type);

    auto* Q_r = ggml_view_3d(ctx, Q,
                             head_dim, n_q_heads, n_tokens,
                             /*nb1=*/Q->nb[1], /*nb2=*/Q->nb[2],
                             /*offset=*/static_cast<std::size_t>(qo_start) * Q_stride_ne2);
    auto* Q_r_perm = ggml_permute(ctx, Q_r, 0, 2, 1, 3);

    auto* K_gather = ggml_get_rows(ctx, k_cached, gather_idx_r);
    auto* V_gather = ggml_get_rows(ctx, v_cached, gather_idx_r);

    auto* K_r = ggml_reshape_3d(ctx, K_gather, head_dim, n_kv_heads, n_kv);
    auto* V_r = ggml_reshape_3d(ctx, V_gather, head_dim, n_kv_heads, n_kv);
    auto* K_r_perm = ggml_permute(ctx, K_r, 0, 2, 1, 3);
    auto* V_r_perm = ggml_permute(ctx, V_r, 0, 2, 1, 3);

    auto* attn = ggml_flash_attn_ext(ctx, Q_r_perm, K_r_perm, V_r_perm,
                                     mask_r, kq_scale,
                                     /*max_bias=*/0.0f,
                                     /*logit_softcap=*/attn_softcap);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    if (sinks) {
        // Sinks must be F32 (asserted by ggml_flash_attn_ext_add_sinks).
        // P5 hoists the BF16→F32 conversion to load time so the cast
        // here is normally a no-op; keep the ternary as a safety net.
        auto* sinks_f32 = (sinks->type == GGML_TYPE_F32)
            ? sinks
            : ggml_cast(ctx, sinks, GGML_TYPE_F32);
        ggml_flash_attn_ext_add_sinks(attn, sinks_f32);
    }
    return ggml_cont(ctx, attn);
}

ggml_tensor* concat_per_request_attn(
    ggml_context* ctx,
    const std::vector<ggml_tensor*>& per_req,
    std::int32_t  head_dim,
    std::int32_t  n_q_heads,
    std::int32_t  n_total) {
    ggml_tensor* attn_concat = per_req[0];
    for (std::size_t r = 1; r < per_req.size(); ++r) {
        attn_concat = ggml_concat(ctx, attn_concat, per_req[r], /*dim=*/2);
    }
    return ggml_reshape_2d(ctx, ggml_cont(ctx, attn_concat),
                           head_dim * n_q_heads, n_total);
}

ggml_tensor* apply_lora_delta(ggml_context* ctx,
                              ggml_tensor*  y,
                              ggml_tensor*  a,
                              ggml_tensor*  b,
                              ggml_tensor*  x,
                              float         scale) {
    if (!a || !b) return y;
    auto* a_out = ggml_mul_mat(ctx, a, x);  // [rank, n_total]
    auto* b_out = ggml_mul_mat(ctx, b, a_out);  // [out, n_total]
    if (scale != 1.0f) {
        b_out = ggml_scale(ctx, b_out, scale);
    }
    return ggml_add(ctx, y, b_out);
}

void upload_graph_inputs(const GraphResult& g,
                         const Executor::BatchPlan& plan) {
    ggml_backend_tensor_set(g.in.tok_input, plan.tokens_i32.data(), 0,
                            plan.tokens_i32.size() * sizeof(std::int32_t));

    // Vision side-inputs (multimodal). The vis_* tensors exist only when the
    // batch carried image spans (declare_graph_inputs allocated them).
    if (g.in.vis_pixels) {
        ggml_backend_tensor_set(g.in.vis_pixels, plan.vis_pixels.data(), 0,
                                plan.vis_pixels.size() * sizeof(float));
        ggml_backend_tensor_set(g.in.vis_pos_embed, plan.vis_pos_embed.data(), 0,
                                plan.vis_pos_embed.size() * sizeof(float));
        ggml_backend_tensor_set(g.in.vis_rope_cos, plan.vis_rope_cos.data(), 0,
                                plan.vis_rope_cos.size() * sizeof(float));
        ggml_backend_tensor_set(g.in.vis_rope_sin, plan.vis_rope_sin.data(), 0,
                                plan.vis_rope_sin.size() * sizeof(float));
        ggml_backend_tensor_set(g.in.vis_img_rows, plan.vis_img_rows.data(), 0,
                                plan.vis_img_rows.size() * sizeof(std::int64_t));
        if (g.in.vis_img_rows_i32) {
            std::vector<std::int32_t> rows32(plan.vis_img_rows.size());
            for (std::size_t i = 0; i < rows32.size(); ++i)
                rows32[i] = static_cast<std::int32_t>(plan.vis_img_rows[i]);
            ggml_backend_tensor_set(g.in.vis_img_rows_i32, rows32.data(), 0,
                                    rows32.size() * sizeof(std::int32_t));
        }
        if (g.in.vis_pool_matrix && !plan.vis_pool_matrix.empty()) {
            ggml_backend_tensor_set(g.in.vis_pool_matrix, plan.vis_pool_matrix.data(),
                                    0, plan.vis_pool_matrix.size() * sizeof(float));
        }
        if (g.in.vis_attn_mask && !plan.vis_attn_mask.empty()) {
            ggml_backend_tensor_set(g.in.vis_attn_mask, plan.vis_attn_mask.data(),
                                    0, plan.vis_attn_mask.size() * sizeof(float));
        }
    }

    // Audio side-inputs (Gemma-4 Conformer). One encoder per clip; aud_pe shared.
    if (g.in.aud_pe && !plan.aud_clips.empty()) {
        ggml_backend_tensor_set(g.in.aud_pe, plan.aud_pe.data(), 0,
                                plan.aud_pe.size() * sizeof(float));
        for (std::size_t c = 0; c < plan.aud_clips.size(); ++c) {
            const auto& clip = plan.aud_clips[c];
            ggml_backend_tensor_set(g.in.aud_features[c], clip.features.data(), 0,
                                    clip.features.size() * sizeof(float));
            ggml_backend_tensor_set(g.in.aud_win_mask[c], clip.win_mask.data(), 0,
                                    clip.win_mask.size() * sizeof(float));
            ggml_backend_tensor_set(g.in.aud_rows[c], clip.rows.data(), 0,
                                    clip.rows.size() * sizeof(std::int64_t));
        }
    }

    // mrope pos_input layouts (both are 4× wide). ggml_rope_multi reads, for a
    // token i and axis-section s, pos[s * ne2 + i] where ne2 is the rope op's
    // token count. Two consumers differ in that count:
    //   * Qwen3-VL (graph_qwen3.cpp) ropes the WHOLE batch at once (ne2 = N), so
    //     the layout is GLOBAL axis-major: [all t | all h | all w | all e]. The
    //     per-token [t,h,w] come from plan.mrope_positions_i32 (image/video rows
    //     carry spatial/temporal positions; text rows are [p,p,p]); the unused
    //     4th axis (section 0) gets the 1-D position.
    //   * Qwen 3.5/3.6 (graph_qwen3_5.cpp) ropes PER REQUEST (ne2 = n_tokens) on
    //     a slice at qo_start*4, so the layout is PER-REQUEST axis-major and all
    //     4 axes equal the token's text position (no spatial mrope).
    const std::int64_t pos_n_expected =
        static_cast<std::int64_t>(plan.positions_i32.size());
    if (!plan.mrope_positions_i32.empty() &&
        g.in.pos_input->ne[0] == pos_n_expected * 4) {
        const std::size_t N = static_cast<std::size_t>(pos_n_expected);
        std::vector<std::int32_t> mrope_pos(N * 4);
        for (std::size_t i = 0; i < N; ++i) {
            mrope_pos[0 * N + i] = plan.mrope_positions_i32[3 * i + 0];  // t
            mrope_pos[1 * N + i] = plan.mrope_positions_i32[3 * i + 1];  // h
            mrope_pos[2 * N + i] = plan.mrope_positions_i32[3 * i + 2];  // w
            mrope_pos[3 * N + i] = plan.positions_i32[i];                // unused
        }
        ggml_backend_tensor_set(g.in.pos_input, mrope_pos.data(), 0,
                                mrope_pos.size() * sizeof(std::int32_t));
    } else if (g.in.pos_input->ne[0] == pos_n_expected * 4) {
        std::vector<std::int32_t> mrope_pos(
            static_cast<std::size_t>(pos_n_expected) * 4);
        std::size_t out_off = 0;
        for (const auto& rp : plan.reqs) {
            const std::int32_t* p_src =
                plan.positions_i32.data() + rp.qo_start;
            for (int axis = 0; axis < 4; ++axis) {
                std::memcpy(mrope_pos.data() + out_off, p_src,
                            static_cast<std::size_t>(rp.n_tokens) *
                            sizeof(std::int32_t));
                out_off += static_cast<std::size_t>(rp.n_tokens);
            }
        }
        ggml_backend_tensor_set(g.in.pos_input, mrope_pos.data(), 0,
                                mrope_pos.size() * sizeof(std::int32_t));
    } else {
        ggml_backend_tensor_set(g.in.pos_input, plan.positions_i32.data(), 0,
                                plan.positions_i32.size() * sizeof(std::int32_t));
    }
    ggml_backend_tensor_set(g.in.kv_idxs, plan.kv_idxs_i64.data(), 0,
                            plan.kv_idxs_i64.size() * sizeof(std::int64_t));
    ggml_backend_tensor_set(g.in.out_idx, plan.sampling_pos_i32.data(), 0,
                            plan.sampling_pos_i32.size() * sizeof(std::int32_t));
    if (plan.pure_decode) {
        if (g.in.packed_gather) {
            ggml_backend_tensor_set(g.in.packed_gather,
                                    plan.packed_gather_idxs.data(), 0,
                                    plan.packed_gather_idxs.size() * sizeof(std::int32_t));
        }
        if (g.in.packed_mask) {
            ggml_backend_tensor_set(g.in.packed_mask,
                                    plan.packed_mask_f16.data(), 0,
                                    plan.packed_mask_f16.size() * sizeof(std::uint16_t));
        }
        if (g.in.packed_mask_full) {
            ggml_backend_tensor_set(
                g.in.packed_mask_full,
                plan.packed_mask_full_f16.data(), 0,
                plan.packed_mask_full_f16.size() * sizeof(std::uint16_t));
        }
        if (g.in.page_indices) {
            ggml_backend_tensor_set(
                g.in.page_indices,
                plan.page_indices_i32.data(), 0,
                plan.page_indices_i32.size() * sizeof(std::int32_t));
            ggml_backend_tensor_set(
                g.in.page_indptr,
                plan.page_indptr_i32.data(), 0,
                plan.page_indptr_i32.size() * sizeof(std::int32_t));
            ggml_backend_tensor_set(
                g.in.last_page_lens,
                plan.last_page_lens_i32.data(), 0,
                plan.last_page_lens_i32.size() * sizeof(std::int32_t));
            // Stash host-side indptr for the FlashInfer planner to consume
            // at kernel launch time without a device→host download. The
            // symbol is provided by paged_attn_cuda.cu when GGML_CUDA is
            // enabled; otherwise the call resolves to a stub.
            pie_paged_attn_set_host_indptr(
                plan.page_indptr_i32.data(),
                plan.page_indptr_i32.size());
        }
    } else {
        for (std::size_t r = 0; r < plan.reqs.size(); ++r) {
            const auto& mask = plan.reqs[r].mask_f16;
            const auto& gat  = plan.reqs[r].gather_idxs;
            ggml_backend_tensor_set(g.in.masks[r], mask.data(), 0,
                                    mask.size() * sizeof(std::uint16_t));
            // Phi-3-small blocksparse upload (when present).
            const auto& mask_bs = plan.reqs[r].mask_blocksparse_f16;
            if (!g.in.masks_blocksparse.empty() &&
                g.in.masks_blocksparse[r] != nullptr &&
                !mask_bs.empty()) {
                ggml_backend_tensor_set(
                    g.in.masks_blocksparse[r], mask_bs.data(), 0,
                    mask_bs.size() * sizeof(std::uint16_t));
            }
            ggml_backend_tensor_set(g.in.gather_idxs[r], gat.data(), 0,
                                    gat.size() * sizeof(std::int32_t));
        }
    }
}

void build_sampling_outputs(ggml_context* ctx,
                            ggml_cgraph*  gf,
                            ggml_tensor*  logits,
                            const Executor::BatchPlan& plan,
                            GraphResult&  res) {
    if (plan.all_greedy) {
        // GPU-greedy fast path: argmax along the vocab axis on-device. The
        // caller downloads only n_slots i32 ids; the F32 logits block
        // never becomes a graph output, so gallocr skips its backing
        // buffer.
        ggml_tensor* tokens_out = ggml_argmax(ctx, logits);
        ggml_set_name(tokens_out, "tokens_out");
        ggml_set_output(tokens_out);
        ggml_build_forward_expand(gf, tokens_out);
        res.tokens_out = tokens_out;
    } else if (plan.uniform_top_sample) {
        // GPU non-greedy fast path: temperature-scaled softmax → probs,
        // take top-K indices, gather the K matching probs. The caller
        // downloads only K * n_slots * 8 bytes and finalizes per-slot
        // top-p / min-p / sample host-side.
        const float inv_t = 1.0f / plan.reqs[0].sampler.temperature;
        ggml_tensor* probs = ggml_soft_max_ext(ctx, logits, /*mask=*/nullptr,
                                               /*scale=*/inv_t,
                                               /*max_bias=*/0.0f);
        // probs is [vocab, n_slots]; ggml_top_k yields indices [K, n_slots]
        // sorted descending. Reshape probs to [1, vocab, n_slots] so
        // get_rows gathers along ne[1] → [1, K, n_slots], then flatten to
        // [K, n_slots]. Both reshapes key on the slot count probs->ne[1],
        // which is >= n_req under speculation (see header).
        const std::int64_t vocab   = probs->ne[0];
        const std::int64_t n_slots = probs->ne[1];
        ggml_tensor* top_k_idx = ggml_top_k(ctx, probs, plan.uniform_top_k);
        ggml_tensor* probs_3d  = ggml_reshape_3d(ctx, probs, 1, vocab, n_slots);
        ggml_tensor* gathered  = ggml_get_rows(ctx, probs_3d, top_k_idx);
        ggml_tensor* top_k_probs =
            ggml_reshape_2d(ctx, gathered, plan.uniform_top_k, n_slots);
        ggml_set_name(top_k_idx,   "top_k_idx");
        ggml_set_name(top_k_probs, "top_k_probs");
        ggml_set_output(top_k_idx);
        ggml_set_output(top_k_probs);
        ggml_build_forward_expand(gf, top_k_idx);
        ggml_build_forward_expand(gf, top_k_probs);
        res.top_k_idx   = top_k_idx;
        res.top_k_probs = top_k_probs;
    } else {
        // Slow path: materialize the full [vocab, n_slots] logits block;
        // the host sampler handles every per-slot cut and the draw.
        ggml_set_name(logits, "logits");
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);
        res.logits = logits;
    }
}

}  // namespace pie_portable_driver
