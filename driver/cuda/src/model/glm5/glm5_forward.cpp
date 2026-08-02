#include "model/glm5/glm5_forward.hpp"

#include "model/act_dump.hpp"
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
#include "ops/flashinfer_moe.hpp"
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

// Expert-block size for the device-side aligned MoE path.
// TEMP ablation switch for perf bring-up: PIE_GLM_ABLATE=lmhead,dsa,moe

// The decode GEMV wins only at a single token: it is one warp per output row
// doing scalar FP32 math, against a batched GEMM on tensor cores. Measured on
// glm5.2-mini (output tok/s, GEMV vs batched): c=1 455/421, c=2 739/790,
// c=4 1195/1555, c=8 1706/3062. Override with `PIE_MOE_GEMV_MAX_TOKENS`.
static constexpr int kGlm5MoeGemvMaxTokens = 1;

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

    // Aligned MoE scratch. `routed_blocks` is the worst case: every one of the
    // `routes` rows lands in a distinct expert block, each padded to `block`.
    if (routed_I > 0 && cfg.num_experts > 0) {
        // Sized for both extremes because the block is chosen per forward:
        // the minimum block yields the most blocks, this one the most rows.
        const int block = kernels::moe_aligned_block(routes, cfg.num_experts);
        const int active_expert_cap = std::min(cfg.num_experts, routes);
        const int max_blocks =
            (routes + active_expert_cap * (kernels::kMoeAlignedBlockMin - 1) +
             kernels::kMoeAlignedBlockMin - 1) /
            kernels::kMoeAlignedBlockMin;
        const int aligned_rows =
            std::max(max_blocks * kernels::kMoeAlignedBlockMin,
                     ((routes + active_expert_cap * (block - 1) + block - 1) /
                      block) * block);
        ws.aligned_block_size  = block;
        ws.aligned_max_blocks  = max_blocks;
        ws.aligned_route_ids   = DeviceTensor::allocate(DType::INT32, {aligned_rows});
        ws.aligned_expert_ids  = DeviceTensor::allocate(DType::INT32, {max_blocks});
        ws.aligned_expert_in   = DeviceTensor::allocate(DType::BF16, {aligned_rows, H});
        ws.aligned_gate_up     = DeviceTensor::allocate(DType::BF16, {aligned_rows, 2 * routed_I});
        ws.aligned_act         = DeviceTensor::allocate(DType::BF16, {aligned_rows, routed_I});
        ws.aligned_out         = DeviceTensor::allocate(DType::BF16, {aligned_rows, H});
        const std::int64_t pw =
            static_cast<std::int64_t>(max_blocks) * sizeof(void*) / sizeof(std::int64_t);
        ws.a_gu_ptrs = DeviceTensor::allocate(DType::INT64, {std::max<std::int64_t>(pw, 1)});
        ws.b_gu_ptrs = DeviceTensor::allocate(DType::INT64, {std::max<std::int64_t>(pw, 1)});
        ws.c_gu_ptrs = DeviceTensor::allocate(DType::INT64, {std::max<std::int64_t>(pw, 1)});
        ws.a_dn_ptrs = DeviceTensor::allocate(DType::INT64, {std::max<std::int64_t>(pw, 1)});
        ws.b_dn_ptrs = DeviceTensor::allocate(DType::INT64, {std::max<std::int64_t>(pw, 1)});
        ws.c_dn_ptrs = DeviceTensor::allocate(DType::INT64, {std::max<std::int64_t>(pw, 1)});
    }
    // flashinfer's fused MoE. Sized for the full token budget: unlike qwen3.5
    // this branch also serves prefill, which is where the padded-block batched
    // GEMM wastes the most work (`max_blocks` provisions for worst-case routing
    // skew and every block runs unconditionally).
    if (routed_I > 0 && cfg.num_experts > 0 && Ktop > 0 &&
        ops::flashinfer_cutlass_moe_enabled() && glm5_moe_gate_up_swapped()) {
        ws.cutlass_max_rows = std::min(N, ops::flashinfer_cutlass_moe_max_rows());
        const std::size_t bytes = ops::flashinfer_cutlass_moe_workspace_bytes(
            ops::MoeActivation::Swiglu, ws.cutlass_max_rows, H, routed_I,
            cfg.num_experts, Ktop, /*tp_size=*/1, /*tp_rank=*/0);
        if (bytes > 0) {
            ws.cutlass_ws = DeviceTensor::allocate(
                DType::UINT8, {static_cast<std::int64_t>(bytes)});
            ws.cutlass_row_map = DeviceTensor::allocate(
                DType::INT32, {static_cast<std::int64_t>(ws.cutlass_max_rows) * Ktop});
        }
    }
    ws.logits        = DeviceTensor::allocate(DType::BF16, {O, cfg.vocab_size});
    return ws;
}

std::size_t glm5_workspace_bytes(
    const HfConfig& cfg,
    int max_tokens,
    int max_logit_rows,
    int max_position_embeddings,
    int tp_size) {
    ScopedDeviceAllocationCounter counter;
    {
        auto workspace = Glm5Workspace::allocate(
            cfg, max_tokens, max_logit_rows,
            max_position_embeddings, tp_size);
    }
    return counter.allocated_bytes();
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
    const std::uint8_t* row_valid_d,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows,
    const StageHooks* hooks)
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
    //
    // GLM-5.2 only ships indexer weights for the layers whose `indexer_types`
    // entry is "full"; a "shared" layer reuses the selection computed by the
    // most recent "full" layer (upstream calls this the index cache — one
    // `topk_indices_buffer` written by indexer layers and read by the skipped
    // ones). Hoisting the mask out of the layer loop gives exactly that.
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

    // Carried across layers: "shared" indexer layers reuse the last mask.
    const std::uint8_t* idx_mask_ptr = nullptr;
    int idx_mask_stride = 0;

    act_dump_step_begin(stream);
    act_dump_bf16("embed", ws.y.data(), total_tokens, H, stream);

    // A layer's closing `y += moe_out` and the next layer's pre-norm are two
    // launches over the same row that neither reads nor writes anything in
    // between, and at decode both cost about what an empty kernel costs. Carry
    // the addend forward and let the pre-norm do it. Held back when the
    // activation dumper is on, which wants `y` finalised at the layer boundary.
    const void* pending_residual = nullptr;

    for (int li = 0; li < cfg.num_hidden_layers; ++li) {
        const auto& Lw = w.layers[static_cast<std::size_t>(li)];

        // ── MLA attention ────────────────────────────────────────────
        if (pending_residual != nullptr) {
            kernels::launch_residual_add_rmsnorm_bf16(
                ws.y.data(), pending_residual, Lw.attn_norm->data(),
                ws.norm_x.data(), total_tokens, H, eps, stream);
            pending_residual = nullptr;
        } else {
            kernels::launch_rmsnorm_bf16(
                ws.y.data(), Lw.attn_norm->data(), ws.norm_x.data(),
                total_tokens, H, eps, stream);
        }

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
            hooks,
            StageHookPoint::OnAttnProj, ws.q_b.data(),
            static_cast<std::uint32_t>(total_tokens),
            static_cast<std::uint32_t>(heads * (q_nope + q_rope)),
            static_cast<std::uint32_t>(li), stream);

        // ── DSA lightning-indexer: build top-k mask for this layer ───────
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

        auto layer_view = mla_cache.layer_view(li);
        const bool fuse_prepare =
            kernels::mla_prepare_supported(q_rope) && !act_dump_enabled();
        if (fuse_prepare) {
            // GLM-5.1+ sets `rope_interleave=true` (config.json), i.e. the
            // GPT-J adjacent-pair convention (dims 2i, 2i+1), not the half/half
            // (NeoX) pairing used by Llama/Kimi. Using the wrong pairing
            // scrambles the rotary subspace for every position > 0.
            kernels::launch_mla_prepare_bf16(
                layer_view,
                ws.kv_a_mqa.data(), Lw.kv_a_norm->data(), ws.q_b.data(),
                ws.kv_c.data(), ws.k_pe.data(),
                ws.q_nope.data(), ws.q_pe.data(),
                positions, qo_indptr, kv_page_indices, kv_page_indptr,
                kv_last_page_lens, total_tokens, num_requests,
                heads, q_nope, eps, cfg.rope_theta, /*interleaved=*/true,
                /*kv_a_row_stride=*/0, /*yarn=*/nullptr, stream, row_valid_d);
        } else {

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

        kernels::launch_write_mla_to_pages(
            layer_view, ws.kv_c.data(), ws.k_pe.data(),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            total_tokens, num_requests, stream, row_valid_d);
        }

        // The kimi_mla kernels read kv_b in BF16, which is what the contract
        // publishes it as -- an FP8 checkpoint is dequantized by the loader.
        const void* kv_b_bf16 = Lw.kv_b_proj->data();
        ops::mla_absorb_q_to_latent_bf16(cublas.handle(),
            ws.q_nope.data(), kv_b_bf16,
            ws.q_nope_latent.data(),
            total_tokens, heads, q_nope, v_dim, kv_lora);

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
        ops::mla_absorb_latent_to_v_bf16(cublas.handle(),
            ws.attn_latent.data(), kv_b_bf16,
            ws.attn_v.data(),
            total_tokens, heads, q_nope, v_dim, kv_lora);
        invoke_stage_hook(
            hooks,
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
        act_dump_bf16(act_dump_layer_tag("post_attn", li).c_str(),
            ws.y.data(), total_tokens, H, stream);
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

        // ── Device-side aligned MoE ─────────────────────────────────
        // vLLM/SGL-style: bucket routes into fixed-size expert blocks on
        // device, run two batched GEMMs, scatter back. No host round-trip and
        // no stream sync, so the forward stays graph-capturable.
        //
        // flashinfer's CUTLASS grouped GEMM does the same job in one call and
        // without the padding: it permutes rows by expert, runs both GEMMs over
        // the *actual* row counts, and its FINALIZE epilogue folds SwiGLU and
        // the top-k weighted sum into GEMM2. The batched path below has to
        // provision `max_blocks` for worst-case routing skew and run every
        // block unconditionally -- 1536 padded rows for 1024 real at E=8, and
        // 17,215 for the same 1024 at the real E=256.
        // At M=1 the routed GEMMs are pure streaming reads with no weight
        // reuse, so the dedicated one-warp-per-row GEMV still beats a grouped
        // GEMM whose tiling assumes an M worth filling -- measured 466 vs 429
        // tok/s on glm5.2-mini at concurrency 1. Reach for the fused runner
        // only above that, where it is unambiguously better.
        const bool gemv_ok =
            Lw.moe_gate_up_proj != nullptr && ws.aligned_block_size > 0 &&
            total_tokens <= ops::moe_gemv_max_tokens(kGlm5MoeGemvMaxTokens) &&
            (H % 8) == 0 && (routed_I % 8) == 0;
        const bool fused_moe_fits =
            !gemv_ok &&
            Lw.moe_gate_up_proj != nullptr && !ws.cutlass_ws.empty() &&
            total_tokens <= ws.cutlass_max_rows &&
            total_tokens >= ops::flashinfer_cutlass_moe_min_rows();
        if (fused_moe_fits &&
            ops::flashinfer_cutlass_moe_bf16(
                ops::MoeActivation::Swiglu,
                static_cast<const std::uint16_t*>(ws.norm_y.data()),
                static_cast<const std::int32_t*>(ws.topk_idx.data()),
                static_cast<const float*>(ws.topk_weights.data()),
                static_cast<const std::uint16_t*>(Lw.moe_gate_up_proj->data()),
                static_cast<const std::uint16_t*>(Lw.moe_down_proj->data()),
                static_cast<std::uint16_t*>(ws.moe_out.data()),
                static_cast<std::uint8_t*>(ws.cutlass_ws.data()),
                static_cast<std::size_t>(ws.cutlass_ws.nbytes()),
                static_cast<std::int32_t*>(ws.cutlass_row_map.data()),
                total_tokens, H, routed_I, E, K,
                /*tp_size=*/1, /*tp_rank=*/0, stream)) {
            // FINALIZE already applied `topk_weights` and summed the K
            // experts, so `ws.moe_out` is complete -- no weighted sum here.
        } else if (gemv_ok) {
            // Decode: at M=1 the routed GEMMs are pure streaming reads with no
            // weight reuse, so one warp per output row beats a batched GEMM
            // whose tiling assumes an M worth filling.
            const int routes = total_tokens * K;
            kernels::launch_moe_gate_up_decode_gemv_bf16(
                static_cast<const std::int32_t*>(ws.topk_idx.data()),
                ws.norm_y.data(), Lw.moe_gate_up_proj->data(),
                ws.aligned_gate_up.data(),
                total_tokens, K, H, routed_I, stream);
            kernels::launch_chunked_swiglu_bf16(
                ws.aligned_gate_up.data(), ws.aligned_act.data(),
                routes, routed_I, stream,
                /*gate_second=*/glm5_moe_gate_up_swapped());
            kernels::launch_moe_down_decode_gemv_bf16(
                static_cast<const std::int32_t*>(ws.topk_idx.data()),
                ws.aligned_act.data(), Lw.moe_down_proj->data(),
                ws.expert_out.data(),
                total_tokens, K, H, routed_I, stream);
            kernels::launch_token_batched_weighted_sum_bf16(
                ws.moe_out.data(), ws.expert_out.data(),
                static_cast<const float*>(ws.topk_weights.data()),
                total_tokens, K, H, stream);
        } else if (Lw.moe_gate_up_proj != nullptr && ws.aligned_block_size > 0) {
            const int routes = total_tokens * K;
            const int block = std::min(ws.aligned_block_size,
                                       kernels::moe_aligned_block(routes, E));
            const int active_expert_cap = std::min(E, routes);
            const int max_blocks =
                (routes + active_expert_cap * (block - 1) + block - 1) / block;
            const int aligned_rows = max_blocks * block;
            if (max_blocks > ws.aligned_max_blocks ||
                aligned_rows > static_cast<int>(ws.aligned_expert_in.shape()[0])) {
                throw std::runtime_error("glm5: aligned MoE scratch too small");
            }

            kernels::launch_moe_align_decode(
                static_cast<const std::int32_t*>(ws.topk_idx.data()),
                static_cast<std::int32_t*>(ws.aligned_route_ids.data()),
                static_cast<std::int32_t*>(ws.aligned_expert_ids.data()),
                /*route_to_aligned_row=*/nullptr,
                routes, E, block, max_blocks, stream);
            kernels::launch_gather_moe_aligned_inputs_bf16(
                ws.norm_y.data(),
                static_cast<const std::int32_t*>(ws.aligned_route_ids.data()),
                ws.aligned_expert_in.data(),
                routes, aligned_rows, K, H,
                /*shared_row_begin=*/-1, total_tokens, stream);
            kernels::launch_build_moe_ptrs_aligned_bf16(
                static_cast<const std::int32_t*>(ws.aligned_expert_ids.data()),
                Lw.moe_gate_up_proj->data(), Lw.moe_down_proj->data(),
                ws.aligned_expert_in.data(), ws.aligned_gate_up.data(),
                ws.aligned_act.data(), ws.aligned_out.data(),
                reinterpret_cast<const void**>(ws.a_gu_ptrs.data()),
                reinterpret_cast<const void**>(ws.b_gu_ptrs.data()),
                reinterpret_cast<void**>(ws.c_gu_ptrs.data()),
                reinterpret_cast<const void**>(ws.a_dn_ptrs.data()),
                reinterpret_cast<const void**>(ws.b_dn_ptrs.data()),
                reinterpret_cast<void**>(ws.c_dn_ptrs.data()),
                max_blocks, block, H, routed_I,
                /*routed_blocks=*/max_blocks, nullptr, nullptr, stream);

            ops::gemm_batched_act_x_wt_bf16(cublas.handle(),
                reinterpret_cast<const void* const*>(ws.b_gu_ptrs.data()),
                reinterpret_cast<const void* const*>(ws.a_gu_ptrs.data()),
                reinterpret_cast<void* const*>(ws.c_gu_ptrs.data()),
                block, 2 * routed_I, H, max_blocks);
            kernels::launch_chunked_swiglu_bf16(
                ws.aligned_gate_up.data(), ws.aligned_act.data(),
                aligned_rows, routed_I, stream,
                /*gate_second=*/glm5_moe_gate_up_swapped());
            ops::gemm_batched_act_x_wt_bf16(cublas.handle(),
                reinterpret_cast<const void* const*>(ws.b_dn_ptrs.data()),
                reinterpret_cast<const void* const*>(ws.a_dn_ptrs.data()),
                reinterpret_cast<void* const*>(ws.c_dn_ptrs.data()),
                block, H, routed_I, max_blocks);
            kernels::launch_reorder_moe_aligned_output_bf16(
                ws.aligned_out.data(),
                static_cast<const std::int32_t*>(ws.aligned_route_ids.data()),
                ws.expert_out.data(),
                routes, aligned_rows, H,
                /*shared_row_begin=*/-1, total_tokens, nullptr, stream);
            kernels::launch_token_batched_weighted_sum_bf16(
                ws.moe_out.data(), ws.expert_out.data(),
                static_cast<const float*>(ws.topk_weights.data()),
                total_tokens, K, H, stream);
        } else {

        // ── Per-expert prefill MoE ──────────────────────────────────
        // Fallback for quantised checkpoints, where the loader keeps the
        // per-expert layout. Builds host-side routing, then walks experts
        // sequentially.
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
        }  // end per-expert fallback

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
        if (!act_dump_enabled()) {
            pending_residual = ws.moe_out.data();
        } else {
            kernels::launch_residual_add_bf16(
                ws.y.data(), ws.moe_out.data(),
                static_cast<std::size_t>(total_tokens) * H, stream);
        }
        act_dump_bf16(act_dump_layer_tag("out", li).c_str(),
            ws.y.data(), total_tokens, H, stream);
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
    if (pending_residual != nullptr && compact_logits) {
        // The gather reads `y`, so it has to be finalised first.
        kernels::launch_residual_add_bf16(
            ws.y.data(), pending_residual,
            static_cast<std::size_t>(total_tokens) * H, stream);
        pending_residual = nullptr;
    }
    if (compact_logits) {
        kernels::launch_gather_bf16_rows(
            static_cast<const std::uint16_t*>(ws.y.data()),
            logit_row_indices_d,
            static_cast<std::uint16_t*>(ws.norm_x.data()),
            num_logit_rows, H, stream);
        final_in = ws.norm_x.data();
    }
    if (pending_residual != nullptr) {
        kernels::launch_residual_add_rmsnorm_bf16(
            ws.y.data(), pending_residual, w.final_norm->data(),
            ws.norm_y.data(), rows, H, eps, stream);
        pending_residual = nullptr;
    } else {
        kernels::launch_rmsnorm_bf16(
            final_in, w.final_norm->data(), ws.norm_y.data(),
            rows, H, eps, stream);
    }
    if (w.lm_head_tp_sharded) {
        throw std::runtime_error(
            "glm5: sharded lm_head not supported in first-pass forward");
    }
    act_dump_bf16("final_norm", ws.norm_y.data(), rows, H, stream);
    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_y.data(), *w.lm_head, logits_out,
        rows, V, H);
    act_dump_bf16("logits", logits_out, rows, V, stream);
}

}  // namespace pie_cuda_driver::model
