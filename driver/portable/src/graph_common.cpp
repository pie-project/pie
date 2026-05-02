#include "graph_common.hpp"

#include <cstring>
#include <string>

#include <ggml-backend.h>

#include "plan.hpp"  // MASK_PAD

namespace pie_portable_driver {

GraphInputs declare_graph_inputs(ggml_context* ctx,
                                 const ForwardEngine::BatchPlan& plan,
                                 std::int32_t n_total,
                                 std::int32_t n_req) {
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

        // Optional no-SWA companion (Gemma 4 full-attention layers).
        if (!plan.packed_mask_full_f16.empty()) {
            in.packed_mask_full = ggml_new_tensor_4d(
                ctx, GGML_TYPE_F16, plan.max_n_kv,
                static_cast<std::int64_t>(MASK_PAD), 1, n_req);
            ggml_set_name(in.packed_mask_full, "kq_mask.packed.full");
            ggml_set_input(in.packed_mask_full);
        }
    } else {
        in.masks.reserve(n_req);
        in.gather_idxs.reserve(n_req);
        for (std::int32_t r = 0; r < n_req; ++r) {
            const auto& R = plan.reqs[r];
            auto* m = ggml_new_tensor_4d(
                ctx, GGML_TYPE_F16, R.n_kv, R.n_tokens_pad, 1, 1);
            ggml_set_name(m, ("kq_mask." + std::to_string(r)).c_str());
            ggml_set_input(m);
            in.masks.push_back(m);

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
                         const ForwardEngine::BatchPlan& plan) {
    ggml_backend_tensor_set(g.in.tok_input, plan.tokens_i32.data(), 0,
                            plan.tokens_i32.size() * sizeof(std::int32_t));
    // Qwen 3.5 / 3.6 use mrope: ggml_rope_multi expects 4 positions per
    // token (t, h, w, +1). The per-request slicing in graph_qwen3_5.cpp
    // takes a contiguous slice [qo_start*4 .. qo_start*4 + n_tokens*4],
    // so the layout must be PER-REQUEST axis-major (not batch-axis-major):
    //   request r contributes [t_0..t_{n-1}, h_0..h_{n-1},
    //                          w_0..w_{n-1}, e_0..e_{n-1}].
    // For text-only inference all 4 axes equal the token's text position.
    const std::int64_t pos_n_expected =
        static_cast<std::int64_t>(plan.positions_i32.size());
    if (g.in.pos_input->ne[0] == pos_n_expected * 4) {
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
        ggml_backend_tensor_set(g.in.packed_gather,
                                plan.packed_gather_idxs.data(), 0,
                                plan.packed_gather_idxs.size() * sizeof(std::int32_t));
        ggml_backend_tensor_set(g.in.packed_mask,
                                plan.packed_mask_f16.data(), 0,
                                plan.packed_mask_f16.size() * sizeof(std::uint16_t));
        if (g.in.packed_mask_full) {
            ggml_backend_tensor_set(
                g.in.packed_mask_full,
                plan.packed_mask_full_f16.data(), 0,
                plan.packed_mask_full_f16.size() * sizeof(std::uint16_t));
        }
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

}  // namespace pie_portable_driver
