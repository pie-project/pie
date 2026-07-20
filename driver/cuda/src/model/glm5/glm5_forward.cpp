#include "model/glm5/glm5_forward.hpp"
#include "model/stage_hooks.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/dsa_indexer.hpp"
#include "kernels/embed.hpp"
#include "kernels/gather_rows.hpp"
#include "kernels/kimi_mla.hpp"
#include "kernels/mla_paged.hpp"
#include "kernels/moe_dispatch.hpp"
#include "kernels/residual_add.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include "kernels/swiglu.hpp"
#include "model/llama_like/qwen3.hpp"  // for make_weight_view

namespace pie_cuda_driver::model {

namespace {

// Build a WeightView for an expert weight, handling the MXFP4 case where
// the weight tensor is stored as a packed byte buffer (1-D UINT8) and the
// QuantMeta carries PerGroup E8M0 scales. Falls back to make_weight_view
// for ordinary BF16 / FP8 weights.
ops::WeightView make_expert_weight_view(
    const DeviceTensor* w,
    const std::optional<QuantMeta>& meta)
{
    if (meta.has_value() && meta->scale != nullptr &&
        meta->kind == QuantMeta::Kind::PerGroup &&
        meta->group_size == 32 &&
        meta->scale->dtype() == DType::UINT8) {
        // MXFP4 expert weight: bytes are nibble-packed and the scale is
        // E8M0 (uint8). Override the dtype so the GEMM dispatcher routes
        // to the MXFP4_PACKED path.
        ops::WeightView v;
        v.data = w->data();
        v.dtype = DType::MXFP4_PACKED;
        v.nbytes = w->nbytes();
        v.scale_data = meta->scale->data();
        v.scale_dtype = DType::UINT8;
        v.scale_numel = meta->scale->numel();
        v.quant_kind = QuantMeta::Kind::PerGroup;
        v.group_size = 32;
        v.channel_axis = meta->channel_axis;
        return v;
    }
    return make_weight_view(w, meta);
}

struct ExpertRouting {
    std::vector<std::vector<std::int32_t>> token_idx;
    std::vector<std::vector<float>> weights;
};

ExpertRouting build_routing(
    const std::vector<std::int32_t>& topk_idx,
    const std::vector<float>& topk_w,
    int N,
    int K,
    int E)
{
    ExpertRouting r;
    r.token_idx.resize(static_cast<std::size_t>(E));
    r.weights.resize(static_cast<std::size_t>(E));
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            const int e = topk_idx[static_cast<std::size_t>(n) * K + k];
            if (e < 0 || e >= E) continue;
            r.token_idx[static_cast<std::size_t>(e)].push_back(n);
            r.weights[static_cast<std::size_t>(e)].push_back(
                topk_w[static_cast<std::size_t>(n) * K + k]);
        }
    }
    return r;
}

}  // namespace

Glm5Workspace Glm5Workspace::allocate(
    const HfConfig& cfg,
    int max_tokens,
    int max_logit_rows,
    int max_position_embeddings,
    int tp_size)
{
    (void)max_position_embeddings;  // reserved for future DSA workspace
    const int T = std::max(1, tp_size);
    const int N = std::max(1, max_tokens);
    const int O = std::max(1, max_logit_rows > 0 ? max_logit_rows : max_tokens);
    const int H = cfg.hidden_size;
    const int local_heads = cfg.num_attention_heads / T;
    const int q_nope = cfg.qk_nope_head_dim;
    const int q_rope = cfg.qk_rope_head_dim;
    const int v_dim = cfg.v_head_dim;
    const int q_lora = cfg.q_lora_rank;
    const int kv_lora = cfg.kv_lora_rank;
    const int dense_I =
        cfg.intermediate_size > 0 ? cfg.intermediate_size / T : 0;
    const int routed_I =
        cfg.moe_intermediate_size > 0 ? cfg.moe_intermediate_size / T : 0;
    const int shared_I =
        cfg.n_shared_experts > 0 && cfg.moe_intermediate_size > 0
            ? cfg.n_shared_experts * cfg.moe_intermediate_size / T
            : 0;
    const int max_I = std::max(1, std::max(dense_I, routed_I));
    const int Ktop = std::max(1, cfg.num_experts_per_tok);
    const int routes = N * Ktop;

    if (H <= 0 || local_heads <= 0 || q_nope <= 0 || q_rope <= 0 ||
        v_dim <= 0 || q_lora <= 0 || kv_lora <= 0) {
        throw std::runtime_error("glm5: cannot allocate workspace with unset dimensions");
    }

    Glm5Workspace ws;
    ws.y             = DeviceTensor::allocate(DType::BF16, {N, H});
    ws.norm_x        = DeviceTensor::allocate(DType::BF16, {N, H});
    ws.q_a           = DeviceTensor::allocate(DType::BF16, {N, q_lora});
    ws.q_b           = DeviceTensor::allocate(DType::BF16, {N, local_heads * (q_nope + q_rope)});
    ws.q_nope        = DeviceTensor::allocate(DType::BF16, {N, local_heads * q_nope});
    ws.kv_a_mqa      = DeviceTensor::allocate(DType::BF16, {N, kv_lora + q_rope});
    ws.kv_c          = DeviceTensor::allocate(DType::BF16, {N, kv_lora});
    ws.k_pe          = DeviceTensor::allocate(DType::BF16, {N, q_rope});
    ws.q_nope_latent = DeviceTensor::allocate(DType::BF16, {N, local_heads * kv_lora});
    ws.q_pe          = DeviceTensor::allocate(DType::BF16, {N, local_heads * q_rope});
    ws.attn_latent   = DeviceTensor::allocate(DType::BF16, {N, local_heads * kv_lora});
    ws.attn_v        = DeviceTensor::allocate(DType::BF16, {N, local_heads * v_dim});
    // DSA lightning-indexer scratch (prefill top-k). Sized from config; inert
    // when the model has no indexer.
    if (cfg.index_n_heads > 0 && cfg.index_head_dim > 0) {
        const int ih = cfg.index_n_heads;
        const int id = cfg.index_head_dim;
        ws.idx_q    = DeviceTensor::allocate(DType::BF16, {N, ih * id});
        ws.idx_k    = DeviceTensor::allocate(DType::BF16, {N, id});
        ws.idx_w    = DeviceTensor::allocate(DType::BF16, {N, ih});
        ws.idx_mask = DeviceTensor::allocate(DType::UINT8,
                          {static_cast<std::int64_t>(N) * N});
    }
    ws.norm_y        = DeviceTensor::allocate(DType::BF16, {N, H});
    ws.gate          = DeviceTensor::allocate(DType::BF16, {N, max_I});
    ws.up            = DeviceTensor::allocate(DType::BF16, {N, max_I});
    ws.router_logits = DeviceTensor::allocate(DType::BF16, {N, std::max(1, cfg.num_experts)});
    ws.topk_idx      = DeviceTensor::allocate(DType::INT32, {N, Ktop});
    ws.topk_weights  = DeviceTensor::allocate(DType::FP32, {N, Ktop});
    ws.route_idx     = DeviceTensor::allocate(DType::INT32, {routes});
    ws.route_w       = DeviceTensor::allocate(DType::FP32, {routes});
    // Per-expert prefill scratch sized for the worst case (all tokens
    // routed to a single expert). This is generous but simple, matching
    // Kimi's worst-case allocation policy.
    ws.expert_in     = DeviceTensor::allocate(DType::BF16, {routes, H});
    ws.expert_gate   = DeviceTensor::allocate(DType::BF16, {routes, std::max(1, routed_I)});
    ws.expert_up     = DeviceTensor::allocate(DType::BF16, {routes, std::max(1, routed_I)});
    ws.expert_out    = DeviceTensor::allocate(DType::BF16, {routes, H});
    ws.moe_out       = DeviceTensor::allocate(DType::BF16, {N, H});
    ws.shared_gate   = DeviceTensor::allocate(DType::BF16, {N, std::max(1, shared_I)});
    ws.shared_up     = DeviceTensor::allocate(DType::BF16, {N, std::max(1, shared_I)});
    ws.shared_act    = DeviceTensor::allocate(DType::BF16, {N, std::max(1, shared_I)});
    ws.shared_out    = DeviceTensor::allocate(DType::BF16, {N, H});
    ws.logits        = DeviceTensor::allocate(DType::BF16, {O, cfg.vocab_size});
    return ws;
}

void glm5_forward_paged(
    const Glm5Weights& w,
    const HfConfig& cfg,
    const Glm5ForwardCfg& fwd_cfg,
    KimiPlanState& mla_plan,
    Glm5Workspace& ws,
    MlaCache& mla_cache,
    DsaCache& dsa_cache,
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
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows)
{
    (void)qo_indptr_h;
    (void)kv_page_indptr_h;
    (void)dsa_cache;          // indexer uses a per-forward mask, no cache yet
    const int T = std::max(1, fwd_cfg.tp_size);
    const int H = cfg.hidden_size;
    const int V = cfg.vocab_size;
    const int heads = cfg.num_attention_heads / T;
    const int q_lora = cfg.q_lora_rank;
    const int kv_lora = cfg.kv_lora_rank;
    const int q_nope = cfg.qk_nope_head_dim;
    const int q_rope = cfg.qk_rope_head_dim;
    const int v_dim = cfg.v_head_dim;
    const int dense_I = cfg.intermediate_size / T;
    const int routed_I = cfg.moe_intermediate_size / T;
    const int shared_I = (cfg.n_shared_experts > 0 && cfg.moe_intermediate_size > 0)
        ? cfg.n_shared_experts * cfg.moe_intermediate_size / T
        : 0;
    const int E = cfg.num_experts;
    const int K = cfg.num_experts_per_tok;
    const float eps = cfg.rms_norm_eps;
    cudaStream_t stream = cublas.stream();
    NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;

    // ── DSA lightning-indexer (prefill top-k) ────────────────────────────
    // The indexer selects the top-`index_topk` keys per query for the main
    // MLA. We currently build the mask only for single-request pure prefill
    // (all keys in-batch; no decode-time indexer cache yet). For seq_len <=
    // index_topk it reduces to dense, and for decode / multi-request it's
    // skipped (dense), which is exact while seq_len <= index_topk.
    const int idx_nh = cfg.index_n_heads;
    const int idx_hd = cfg.index_head_dim;
    const int idx_topk = cfg.index_topk;
    const bool use_indexer =
        idx_nh > 0 && idx_hd > 0 && idx_topk > 0 &&
        !is_pure_decode && num_requests == 1 &&
        w.layers[0].idx_wq_b != nullptr && !ws.idx_mask.empty();

    // ── Token embedding ──────────────────────────────────────────────
    if (w.embed_tp_sharded) {
        if (tp == nullptr) {
            throw std::runtime_error("glm5: sharded embed requires TP communicator");
        }
        kernels::launch_embed_bf16_vocab_shard(
            token_ids, w.embed->data(), ws.y.data(),
            total_tokens, H, static_cast<int>(w.embed->shape()[0]),
            w.embed_tp_vocab_offset, stream);
        tp->all_reduce_bf16(ws.y.data(),
            static_cast<std::size_t>(total_tokens) * static_cast<std::size_t>(H),
            ncclSum, stream);
    } else {
        kernels::launch_embed_bf16(
            token_ids, w.embed->data(), ws.y.data(),
            total_tokens, H, cfg.vocab_size, stream);
    }

    for (int li = 0; li < cfg.num_hidden_layers; ++li) {
        const auto& Lw = w.layers[static_cast<std::size_t>(li)];

        // ── MLA attention ────────────────────────────────────────────
        kernels::launch_rmsnorm_bf16(
            ws.y.data(), Lw.attn_norm->data(), ws.norm_x.data(),
            total_tokens, H, eps, stream);

        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(),
            make_weight_view(Lw.q_a_proj, Lw.q_a_proj_quant),
            ws.q_a.data(), total_tokens, q_lora, H);
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(),
            make_weight_view(Lw.kv_a_proj_with_mqa, Lw.kv_a_proj_with_mqa_quant),
            ws.kv_a_mqa.data(), total_tokens, kv_lora + q_rope, H);

        kernels::launch_rmsnorm_bf16(
            ws.q_a.data(), Lw.q_a_norm->data(), ws.q_a.data(),
            total_tokens, q_lora, eps, stream);
        ops::gemm_act_x_w(cublas.handle(),
            ws.q_a.data(),
            make_weight_view(Lw.q_b_proj, Lw.q_b_proj_quant),
            ws.q_b.data(), total_tokens, heads * (q_nope + q_rope), q_lora);
        invoke_stage_hook(
            StageHookPoint::OnAttnProj, ws.q_b.data(),
            static_cast<std::uint32_t>(total_tokens),
            static_cast<std::uint32_t>(heads * (q_nope + q_rope)),
            static_cast<std::uint32_t>(li), stream);

        // ── DSA lightning-indexer: build top-k mask for this layer ───────
        const std::uint8_t* idx_mask_ptr = nullptr;
        int idx_mask_stride = 0;
        if (use_indexer && Lw.idx_wq_b != nullptr) {
            // q_idx = wq_b(q_a_normed); k_idx = wk(norm_x); w = weights_proj(norm_x)
            ops::gemm_act_x_w(cublas.handle(),
                ws.q_a.data(),
                make_weight_view(Lw.idx_wq_b, Lw.idx_wq_b_quant),
                ws.idx_q.data(), total_tokens, idx_nh * idx_hd, q_lora);
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_x.data(),
                make_weight_view(Lw.idx_wk, Lw.idx_wk_quant),
                ws.idx_k.data(), total_tokens, idx_hd, H);
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_x.data(),
                make_weight_view(Lw.idx_weights_proj, std::nullopt),
                ws.idx_w.data(), total_tokens, idx_nh, H);
            kernels::launch_dsa_index_knorm_rope_bf16(
                ws.idx_k.data(), Lw.idx_k_norm_weight->data(),
                Lw.idx_k_norm_bias->data(), positions,
                total_tokens, idx_hd, q_rope, cfg.rope_theta, /*eps=*/1e-6f,
                stream);
            kernels::launch_dsa_index_q_rope_bf16(
                ws.idx_q.data(), positions,
                total_tokens, idx_nh, idx_hd, q_rope, cfg.rope_theta, stream);
            kernels::launch_dsa_index_topk_mask(
                ws.idx_q.data(), ws.idx_k.data(), ws.idx_w.data(),
                static_cast<std::uint8_t*>(ws.idx_mask.data()),
                total_tokens, idx_nh, idx_hd, idx_topk, stream);
            idx_mask_ptr = static_cast<const std::uint8_t*>(ws.idx_mask.data());
            idx_mask_stride = total_tokens;
        }

        kernels::launch_kimi_split_kv_a_norm_bf16(
            ws.kv_a_mqa.data(), Lw.kv_a_norm->data(),
            ws.kv_c.data(), ws.k_pe.data(),
            total_tokens, kv_lora, q_rope, eps, stream);

        kernels::launch_kimi_split_q_b_bf16(
            ws.q_b.data(), ws.q_nope.data(), ws.q_pe.data(),
            total_tokens, heads, q_nope, q_rope, stream);

        // RoPE. GLM-5.1 sets `rope_interleave=true` (config.json), i.e. the
        // GPT-J adjacent-pair convention (dims 2i, 2i+1), not the half/half
        // (NeoX) pairing used by Llama/Kimi. Using the wrong pairing scrambles
        // the rotary subspace for every position > 0 and produces degenerate
        // output, so we pass interleaved=true here.
        kernels::launch_rope_bf16(
            ws.q_pe.data(), ws.k_pe.data(), positions,
            total_tokens, heads, 1, q_rope, cfg.rope_theta, stream,
            /*interleaved=*/true);

        auto layer_view = mla_cache.layer_view(li);
        kernels::launch_write_mla_to_pages(
            layer_view, ws.kv_c.data(), ws.k_pe.data(),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            total_tokens, num_requests, stream);

        // kv_b_proj_bf16 holds a BF16 copy of the (possibly FP8) kv_b
        // weight that the kimi_mla kernels require.
        const void* kv_b_bf16 = Lw.kv_b_proj_bf16->data();
        kernels::launch_kimi_q_nope_to_latent_bf16(
            ws.q_nope.data(), kv_b_bf16,
            ws.q_nope_latent.data(),
            total_tokens, heads, q_nope, v_dim, kv_lora, stream);

        if (!mla_plan.mla_plan) {
            throw std::runtime_error("glm5: MLA plan missing; prepare hook did not run");
        }
        ops::dispatch_attention_mla_bf16(
            *mla_plan.mla_plan,
            ws.q_nope_latent.data(),
            ws.q_pe.data(),
            layer_view,
            ws.attn_latent.data(),
            kv_page_indices,
            attn_ws,
            stream,
            /*lse_out=*/nullptr,
            qo_indptr, kv_page_indptr, kv_last_page_lens,
            idx_mask_ptr, idx_mask_stride);
        kernels::launch_kimi_latent_to_v_bf16(
            ws.attn_latent.data(), kv_b_bf16,
            ws.attn_v.data(),
            total_tokens, heads, q_nope, v_dim, kv_lora, stream);
        invoke_stage_hook(
            StageHookPoint::OnAttn, ws.q_b.data(),
            static_cast<std::uint32_t>(total_tokens),
            static_cast<std::uint32_t>(heads * (q_nope + q_rope)),
            static_cast<std::uint32_t>(li), stream);

        if (T == 1) {
            ops::gemm_act_x_w(cublas.handle(),
                ws.attn_v.data(),
                make_weight_view(Lw.o_proj, Lw.o_proj_quant),
                ws.y.data(), total_tokens, H, heads * v_dim, /*beta=*/1.f);
        } else {
            ops::gemm_act_x_w(cublas.handle(),
                ws.attn_v.data(),
                make_weight_view(Lw.o_proj, Lw.o_proj_quant),
                ws.norm_x.data(), total_tokens, H, heads * v_dim);
            tp->all_reduce_bf16(ws.norm_x.data(),
                static_cast<std::size_t>(total_tokens) * H, ncclSum, stream);
            kernels::launch_residual_add_bf16(
                ws.y.data(), ws.norm_x.data(),
                static_cast<std::size_t>(total_tokens) * H, stream);
        }

        // ── MLP / MoE ────────────────────────────────────────────────
        kernels::launch_rmsnorm_bf16(
            ws.y.data(), Lw.mlp_norm->data(), ws.norm_y.data(),
            total_tokens, H, eps, stream);

        if (!Lw.is_moe) {
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_y.data(),
                make_weight_view(Lw.dense_gate_proj, Lw.dense_gate_quant),
                ws.gate.data(), total_tokens, dense_I, H);
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_y.data(),
                make_weight_view(Lw.dense_up_proj, Lw.dense_up_quant),
                ws.up.data(), total_tokens, dense_I, H);
            kernels::launch_swiglu_bf16(
                ws.gate.data(), ws.up.data(), ws.gate.data(),
                total_tokens * dense_I, stream);
            if (T == 1) {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.gate.data(),
                    make_weight_view(Lw.dense_down_proj, Lw.dense_down_quant),
                    ws.y.data(), total_tokens, H, dense_I, /*beta=*/1.f);
            } else {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.gate.data(),
                    make_weight_view(Lw.dense_down_proj, Lw.dense_down_quant),
                    ws.norm_x.data(), total_tokens, H, dense_I);
                tp->all_reduce_bf16(ws.norm_x.data(),
                    static_cast<std::size_t>(total_tokens) * H, ncclSum, stream);
                kernels::launch_residual_add_bf16(
                    ws.y.data(), ws.norm_x.data(),
                    static_cast<std::size_t>(total_tokens) * H, stream);
            }
            continue;
        }

        // ── MoE router ──────────────────────────────────────────────
        // The router is BF16 and quantization-free on GLM-5.1.
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_y.data(), *Lw.router,
            ws.router_logits.data(), total_tokens, E, H);
        // noaux_tc + sigmoid scoring with optional per-expert correction bias.
        kernels::launch_topk_sigmoid_bf16(
            ws.router_logits.data(),
            static_cast<std::int32_t*>(ws.topk_idx.data()),
            static_cast<float*>(ws.topk_weights.data()),
            Lw.e_score_correction_bias != nullptr
                ? static_cast<const float*>(Lw.e_score_correction_bias->data())
                : nullptr,
            total_tokens, E, K, cfg.norm_topk_prob,
            cfg.routed_scaling_factor, stream);

        // ── Per-expert prefill MoE ──────────────────────────────────
        // Single path that works for prefill and decode. Builds host-side
        // routing, then walks experts sequentially. This is the simple
        // worst-case-correct path; Kimi's INT4 decode fast-path doesn't
        // apply to GLM-5.1's FP8 expert layout.
        std::vector<std::int32_t> topk_idx_h(
            static_cast<std::size_t>(total_tokens) * K);
        std::vector<float> topk_w_h(static_cast<std::size_t>(total_tokens) * K);
        CUDA_CHECK(cudaMemcpyAsync(
            topk_idx_h.data(), ws.topk_idx.data(),
            topk_idx_h.size() * sizeof(std::int32_t),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            topk_w_h.data(), ws.topk_weights.data(),
            topk_w_h.size() * sizeof(float),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaMemsetAsync(ws.moe_out.data(), 0,
            static_cast<std::size_t>(total_tokens) * H * sizeof(std::uint16_t),
            stream));

        const auto routing =
            build_routing(topk_idx_h, topk_w_h, total_tokens, K, E);
        for (int e = 0; e < E; ++e) {
            const auto& tok_idx = routing.token_idx[static_cast<std::size_t>(e)];
            const int Ne = static_cast<int>(tok_idx.size());
            if (Ne == 0) continue;
            const auto& wts = routing.weights[static_cast<std::size_t>(e)];
            const auto& Ew = Lw.experts[static_cast<std::size_t>(e)];

            CUDA_CHECK(cudaMemcpyAsync(
                ws.route_idx.data(), tok_idx.data(),
                static_cast<std::size_t>(Ne) * sizeof(std::int32_t),
                cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                ws.route_w.data(), wts.data(),
                static_cast<std::size_t>(Ne) * sizeof(float),
                cudaMemcpyHostToDevice, stream));
            kernels::launch_gather_bf16_rows(
                static_cast<const std::uint16_t*>(ws.norm_y.data()),
                static_cast<const std::int32_t*>(ws.route_idx.data()),
                static_cast<std::uint16_t*>(ws.expert_in.data()),
                Ne, H, stream);
            ops::gemm_act_x_w(cublas.handle(),
                ws.expert_in.data(),
                make_expert_weight_view(Ew.gate_proj, Ew.gate_quant),
                ws.expert_gate.data(), Ne, routed_I, H);
            ops::gemm_act_x_w(cublas.handle(),
                ws.expert_in.data(),
                make_expert_weight_view(Ew.up_proj, Ew.up_quant),
                ws.expert_up.data(), Ne, routed_I, H);
            kernels::launch_swiglu_bf16(
                ws.expert_gate.data(), ws.expert_up.data(),
                ws.expert_gate.data(), Ne * routed_I, stream);
            ops::gemm_act_x_w(cublas.handle(),
                ws.expert_gate.data(),
                make_expert_weight_view(Ew.down_proj, Ew.down_quant),
                ws.expert_out.data(), Ne, H, routed_I);
            kernels::launch_scatter_add_weighted_bf16(
                ws.moe_out.data(), ws.expert_out.data(),
                static_cast<const std::int32_t*>(ws.route_idx.data()),
                static_cast<const float*>(ws.route_w.data()),
                Ne, H, stream);
        }

        // ── Shared experts ──────────────────────────────────────────
        if (shared_I > 0 && Lw.shared_gate_proj != nullptr) {
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_y.data(),
                make_expert_weight_view(Lw.shared_gate_proj, Lw.shared_gate_quant),
                ws.shared_gate.data(), total_tokens, shared_I, H);
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_y.data(),
                make_expert_weight_view(Lw.shared_up_proj, Lw.shared_up_quant),
                ws.shared_up.data(), total_tokens, shared_I, H);
            kernels::launch_swiglu_bf16(
                ws.shared_gate.data(), ws.shared_up.data(),
                ws.shared_act.data(), total_tokens * shared_I, stream);
            ops::gemm_act_x_w(cublas.handle(),
                ws.shared_act.data(),
                make_expert_weight_view(Lw.shared_down_proj, Lw.shared_down_quant),
                ws.shared_out.data(), total_tokens, H, shared_I);
            kernels::launch_residual_add_bf16(
                ws.moe_out.data(), ws.shared_out.data(),
                static_cast<std::size_t>(total_tokens) * H, stream);
        }

        if (T > 1) {
            tp->all_reduce_bf16(ws.moe_out.data(),
                static_cast<std::size_t>(total_tokens) * H, ncclSum, stream);
        }
        kernels::launch_residual_add_bf16(
            ws.y.data(), ws.moe_out.data(),
            static_cast<std::size_t>(total_tokens) * H, stream);
    }

    if (!fwd_cfg.emit_logits) {
        return;
    }

    // ── Final norm + lm_head ─────────────────────────────────────────
    const bool compact_logits =
        logit_row_indices_d != nullptr && num_logit_rows > 0 &&
        num_logit_rows < total_tokens;
    const int rows = compact_logits ? num_logit_rows : total_tokens;
    const void* final_in = ws.y.data();
    if (compact_logits) {
        kernels::launch_gather_bf16_rows(
            static_cast<const std::uint16_t*>(ws.y.data()),
            logit_row_indices_d,
            static_cast<std::uint16_t*>(ws.norm_x.data()),
            num_logit_rows, H, stream);
        final_in = ws.norm_x.data();
    }
    kernels::launch_rmsnorm_bf16(
        final_in, w.final_norm->data(), ws.norm_y.data(),
        rows, H, eps, stream);
    if (w.lm_head_tp_sharded) {
        throw std::runtime_error(
            "glm5: sharded lm_head not supported in first-pass forward");
    }
    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_y.data(), *w.lm_head, logits_out,
        rows, V, H);
}

}  // namespace pie_cuda_driver::model
