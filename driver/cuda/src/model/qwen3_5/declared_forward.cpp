#include "model/qwen3_5/declared_forward.hpp"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/causal_conv1d.hpp"
#include "kernels/deinterleave.hpp"
#include "kernels/embed.hpp"
#include "kernels/gated_delta_net.hpp"
#include "kernels/gather_rows.hpp"
#include "kernels/kv_paged.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include "kernels/split_packed.hpp"
#include "kernels/swiglu.hpp"
#include "ops/attention_flashinfer.hpp"
#include "ops/attention_naive_paged.hpp"
#include "ops/gemm.hpp"

namespace pie_cuda_driver::model {

namespace {

using pie_forward::PieForwardNormVariant;
using pie_forward::PieForwardOp;
using pie_forward::PieForwardOpKind;
using pie_forward::PieForwardRopeKind;

// A plan weight name split into its layer index and field — the llama_like
// executor's parse (`llama_like/declared_forward.cpp`), same contract: a
// name the resolver does not know means the trace and this executor have
// drifted, so it throws rather than half-executing.
struct ParsedWeightName {
    int layer = -1;
    std::string_view field;
};

[[noreturn]] void throw_unknown_weight(std::string_view name) {
    throw std::runtime_error(
        "declared qwen35 forward: unknown weight name '" + std::string(name) +
        "' (trace vocabulary is forward/src/family.rs's)");
}

ParsedWeightName parse_weight_name(std::string_view name) {
    constexpr std::string_view prefix = "layer.";
    if (name.substr(0, prefix.size()) != prefix) {
        return ParsedWeightName{-1, name};
    }
    const std::size_t dot = name.find('.', prefix.size());
    if (dot == std::string_view::npos) throw_unknown_weight(name);
    int layer = -1;
    const char* first = name.data() + prefix.size();
    const char* last = name.data() + dot;
    const auto [ptr, ec] = std::from_chars(first, last, layer);
    if (ec != std::errc() || ptr != last || layer < 0) {
        throw_unknown_weight(name);
    }
    return ParsedWeightName{layer, name.substr(dot + 1)};
}

const Qwen3_5LayerWeights& layer_of(
    const Qwen3_5Weights& w, const ParsedWeightName& nm,
    std::string_view name)
{
    if (nm.layer < 0 || nm.layer >= static_cast<int>(w.layers.size())) {
        throw_unknown_weight(name);
    }
    return w.layers[nm.layer];
}

const DeviceTensor* require(const DeviceTensor* t, std::string_view name) {
    if (t == nullptr) {
        throw std::runtime_error(
            "declared qwen35 forward: weight '" + std::string(name) +
            "' is named by the trace but not bound");
    }
    return t;
}

[[noreturn]] void throw_drift(const std::string& what) {
    throw std::runtime_error(
        "declared qwen35 forward: " + what +
        "; the trace's shape drifted from family.rs's hybrid body");
}

}  // namespace

bool qwen35_declared_exec_trace_enabled() {
    static const bool enabled =
        std::getenv("PIE_DECLARED_FORWARD_TRACE") != nullptr;
    return enabled;
}

void qwen3_5_forward_declared(
    const Qwen35DeclaredPlan& declared,
    const Qwen3_5Weights& w,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    const Qwen3_5PlanState& plan_state,
    Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la,
    KvCache& cache,
    RecurrentStateCache& state_cache,
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
    int total_tokens,
    int num_requests,
    bool is_pure_decode,
    const std::uint32_t* w_page_d,
    const std::uint32_t* w_off_d,
    const std::uint8_t* row_valid_d,
    bool has_write_desc,
    const std::int32_t* slot_ids_h,
    const std::uint8_t* is_fresh_h,
    const std::int32_t* slot_ids_d,
    const std::uint8_t* is_fresh_d,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows)
{
    const pie_forward::ForwardPlan& plan = declared.plan;
    if (qwen35_declared_exec_trace_enabled()) {
        std::fprintf(stderr,
                     "[declared-qwen35-exec] N=%d R=%d decode=%d ops=%zu\n",
                     total_tokens, num_requests, is_pure_decode ? 1 : 0,
                     plan.op_count());
    }
    // The decode slice: the caller's gate admits only pure-decode fires,
    // and every state-op arm below emits the decode-update lowering. A
    // non-decode fire reaching this walk is a caller bug, not a fallback.
    if (!is_pure_decode) {
        throw std::runtime_error(
            "declared qwen35 forward: non-decode fire reached the decode "
            "slice (the caller's gate must exclude it)");
    }

    const int N = total_tokens;
    const int R = num_requests;
    const int H = cfg.hidden_size;
    const int V = cfg.vocab_size;
    const float eps = cfg.rms_norm_eps;
    // TP=1 by the build gate (declared_facts refuses tp>1), so every dim
    // is the unsharded config dim — the hand-written bodies' T==1 case.
    const int num_q_heads = cfg.num_attention_heads;
    const int num_kv_heads = cfg.num_key_value_heads;
    const int d = cfg.head_dim;
    const int Hq = num_q_heads * d;
    const int Hk = num_kv_heads * d;
    const int qgkv_dim = 2 * Hq + 2 * Hk;
    const int I = cfg.intermediate_size;
    const int K_h = cfg.linear_num_key_heads;
    const int V_h = cfg.linear_num_value_heads;
    const int K_d = cfg.linear_key_head_dim;
    const int V_d = cfg.linear_value_head_dim;
    const int K_dim = K_h * K_d;
    const int V_dim = V_h * V_d;
    const int conv_dim = 2 * K_dim + V_dim;
    const int conv_K = cfg.linear_conv_kernel_dim;
    // Inherit cublas's stream so every launch lands on the captured graph
    // (qwen3_5_forward_paged's stream setup, same reasoning).
    cudaStream_t stream = cublas.stream();

    // The hand-written body's explicit-KV-write layout validation, verbatim
    // — same inputs, same throw, so the two paths refuse identically.
    if (has_write_desc) {
        const bool has_full_attention = std::any_of(
            w.layers.begin(), w.layers.end(), [](const auto& layer) {
                return layer.kind == Qwen3_5LayerWeights::Kind::FullAttn;
            });
        if (w_page_d == nullptr || w_off_d == nullptr ||
            !cache.format().is_native_bf16() || !has_full_attention) {
            throw std::runtime_error(
                "Qwen3.5 explicit KV writes are unsupported by this layout");
        }
    }

    // Per-slot reset for freshly (re)assigned rs slots — the hand-written
    // reset stage minus the commit-advance / rs-buffer branches the
    // caller's gate excluded. (On a pure-decode fire the runtime guarantees
    // no slot is fresh — freshness only occurs on a context's first fire, a
    // prefill — but the hand-written body still runs this check, so the
    // walk runs it too rather than reasoning it away.)
    if (slot_ids_h != nullptr && is_fresh_h != nullptr) {
        if (std::any_of(is_fresh_h, is_fresh_h + R,
                        [](auto fresh) { return fresh != 0; })) {
            if (slot_ids_d != nullptr && is_fresh_d != nullptr) {
                state_cache.reset_slots_if_fresh(
                    slot_ids_d, is_fresh_d, R, stream);
            } else {
                for (int r = 0; r < R; ++r) {
                    if (is_fresh_h[r]) {
                        state_cache.reset_slot(slot_ids_h[r], stream);
                    }
                }
            }
        }
    } else if (!is_pure_decode) {
        state_cache.reset(stream);  // unreachable on this slice; mirrored
    }

    // Attention plan pointers, read exactly as qwen3_5_forward_paged reads
    // them (prepare hoisted the host-side planning out of the body).
    const ops::DecodePlanCache* decode_plan =
        plan_state.decode_plan ? plan_state.decode_plan.get() : nullptr;
    const ops::PrefillPlanCache* prefill_plan =
        (plan_state.use_prefill_plan && plan_state.prefill_plan)
            ? plan_state.prefill_plan.get()
            : nullptr;

    // GDN recurrent-state facts, hoisted once (constant across layers).
    const bool state_bf16 = state_cache.recurrent_state_bf16();
    const auto slot_stride = static_cast<long long>(
        state_cache.recurrent_slot_stride_floats());
    // Decode GQA lowering choice — the hand-written
    // `use_decode_gqa_recurrent`, term for term (`linear_decode` is true on
    // this slice: pure decode, rs_buffer_write excluded by the gate).
    const bool use_decode_gqa_recurrent =
        slot_ids_d != nullptr && V_h != K_h && V_h % K_h == 0;
    // What the recurrence consumes when the GQA kernels don't index the
    // compact K_h layout directly (the `q_recur_full` indirection).
    const float* q_recur_full =
        (V_h == K_h) ? la.q_pre.data() : la.q_norm.data();
    const float* k_recur_full =
        (V_h == K_h) ? la.k_pre.data() : la.k_norm.data();

    // Whether the gate_up Matmul took the fused binding; decides which
    // swiglu kernel the following Swiglu op launches (the hand-written
    // fused-vs-unfused pairing in qwen35_dense_mlp_block).
    bool gate_up_used_fused = false;

    const std::size_t op_count = plan.op_count();
    for (std::size_t i = 0; i < op_count; ++i) {
        const PieForwardOp& op = plan.op(i);
        switch (op.kind) {
        case PieForwardOpKind::Embed: {
            const std::string_view name = plan.weight_name(op);
            if (name != "embed") throw_unknown_weight(name);
            kernels::launch_embed_bf16(
                token_ids, require(w.embed, name)->data(), ws.y.data(),
                N, H, cfg.vocab_size, stream);
            break;
        }
        case PieForwardOpKind::Rmsnorm: {
            // The dense hybrid folds Gemma everywhere (declared_facts'
            // norm-variant derivation); a Plain variant here is drift.
            if (op.param0 !=
                static_cast<std::uint32_t>(PieForwardNormVariant::Gemma)) {
                throw_drift("only the Gemma rmsnorm variant is emitted "
                            "(the dense hybrid folds (1+w) everywhere)");
            }
            const std::string_view name = plan.weight_name(op);
            const ParsedWeightName nm = parse_weight_name(name);
            if (nm.field == "attn_norm") {
                const auto& layer = layer_of(w, nm, name);
                kernels::launch_rmsnorm_gemma_bf16(
                    ws.y.data(), require(layer.attn_norm_pre, name)->data(),
                    ws.norm_x.data(), N, H, eps, stream);
            } else if (nm.field == "mlp_norm") {
                // The qwen3_5 MLP reads norm_x (not llama_like's norm_y):
                // qwen3_5_forward_paged's post-attention norm, verbatim.
                const auto& layer = layer_of(w, nm, name);
                kernels::launch_rmsnorm_gemma_bf16(
                    ws.y.data(), require(layer.mlp_norm_pre, name)->data(),
                    ws.norm_x.data(), N, H, eps, stream);
            } else if (nm.layer < 0 && nm.field == "final_norm") {
                // Emitted at its op position: the hand-written epilogue
                // final-norms ALL rows into norm_x first and gathers the
                // compact-logit rows afterwards (norm-then-gather — the
                // opposite interleave from llama_like's epilogue), so the
                // LmHead arm below only gathers and multiplies.
                kernels::launch_rmsnorm_gemma_bf16(
                    ws.y.data(), require(w.final_norm, name)->data(),
                    ws.norm_x.data(), N, H, eps, stream);
            } else {
                throw_unknown_weight(name);
            }
            break;
        }
        case PieForwardOpKind::Matmul: {
            const std::string_view name = plan.weight_name(op);
            const ParsedWeightName nm = parse_weight_name(name);
            const auto& layer = layer_of(w, nm, name);
            const float beta = op.param0 != 0 ? 1.f : 0.f;
            const bool linear =
                layer.kind == Qwen3_5LayerWeights::Kind::LinearAttn;
            // ── GDN in-projections (read norm_x, the pre-attn norm) ──
            if (nm.field == "in_proj_qkvz") {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.norm_x.data(),
                    *require(layer.la_in_proj_qkvz, name),
                    la.mixed_qkvz.data(), N, conv_dim + V_dim, H);
            } else if (nm.field == "in_proj_ba") {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.norm_x.data(),
                    *require(layer.la_in_proj_ba, name),
                    la.ba.data(), N, 2 * V_h, H);
            } else if (nm.field == "in_proj_qkv") {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.norm_x.data(),
                    *require(layer.la_in_proj_qkv, name),
                    la.mixed_qkv.data(), N, conv_dim, H);
            } else if (nm.field == "in_proj_z") {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.norm_x.data(),
                    *require(layer.la_in_proj_z, name),
                    la.z.data(), N, V_dim, H);
            } else if (nm.field == "in_proj_a") {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.norm_x.data(),
                    *require(layer.la_in_proj_a, name),
                    la.a.data(), N, V_h, H);
            } else if (nm.field == "in_proj_b") {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.norm_x.data(),
                    *require(layer.la_in_proj_b, name),
                    la.b.data(), N, V_h, H);
            // ── Full-attention projections ───────────────────────────
            } else if (nm.field == "qgkv") {
                // The trace committed to the fused [2q|k|v] bank; the
                // caller's gate already confirmed the staging buffer holds
                // N rows (the hand-written `use_fused_qgkv` availability
                // check), so a missing bank here is drift, not dispatch.
                ops::gemm_act_x_w(cublas.handle(),
                    ws.norm_x.data(),
                    ops::WeightView(*require(layer.fa_qgkv_proj_fused, name)),
                    ws.gate_up_fused.data(), N, qgkv_dim, H);
            } else if (nm.field == "q_proj") {
                // 2×-wide gated q → the packed [query | gate] buffer.
                ops::gemm_act_x_w(cublas.handle(),
                    ws.norm_x.data(),
                    make_weight_view(require(layer.fa_q_proj, name),
                                     layer.fa_q_proj_quant),
                    la.fa_qg_packed.data(), N, 2 * Hq, H);
            } else if (nm.field == "k_proj") {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.norm_x.data(),
                    make_weight_view(require(layer.fa_k_proj, name),
                                     layer.fa_k_proj_quant),
                    ws.k.data(), N, Hk, H);
            } else if (nm.field == "v_proj") {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.norm_x.data(),
                    make_weight_view(require(layer.fa_v_proj, name),
                                     layer.fa_v_proj_quant),
                    ws.v.data(), N, Hk, H);
            // ── Output projections (residual folded via beta=1) ──────
            } else if (nm.field == "o_proj") {
                if (linear) {
                    ops::gemm_act_x_w(cublas.handle(),
                        la.core_out_bf16.data(),
                        *require(layer.la_out_proj, name),
                        ws.y.data(), N, H, V_dim, beta);
                } else {
                    ops::gemm_act_x_w(cublas.handle(),
                        ws.attn_out.data(),
                        make_weight_view(require(layer.fa_o_proj, name),
                                         layer.fa_o_proj_quant),
                        ws.y.data(), N, H, Hq, beta);
                }
            // ── Dense MLP ────────────────────────────────────────────
            } else if (nm.field == "gate_up") {
                // One traced matmul; whether the binding materialised it
                // fused is this emitter's call — the hand-written
                // qwen35_dense_mlp_block's dispatch, verbatim.
                gate_up_used_fused =
                    layer.gate_up_proj_fused != nullptr &&
                    !ws.gate_up_fused.empty();
                if (gate_up_used_fused) {
                    ops::gemm_act_x_w(cublas.handle(),
                        ws.norm_x.data(),
                        ops::WeightView(*layer.gate_up_proj_fused),
                        ws.gate_up_fused.data(), N, 2 * I, H);
                } else {
                    ops::gemm_act_x_w(cublas.handle(),
                        ws.norm_x.data(),
                        make_weight_view(require(layer.gate_proj, name),
                                         layer.gate_proj_quant),
                        ws.gate.data(), N, I, H);
                    ops::gemm_act_x_w(cublas.handle(),
                        ws.norm_x.data(),
                        make_weight_view(require(layer.up_proj, name),
                                         layer.up_proj_quant),
                        ws.up.data(), N, I, H);
                }
            } else if (nm.field == "down") {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.gate.data(),
                    make_weight_view(require(layer.down_proj, name),
                                     layer.down_proj_quant),
                    ws.y.data(), N, H, I, beta);
            } else {
                throw_unknown_weight(name);
            }
            break;
        }
        case PieForwardOpKind::SplitQkv: {
            // Fused full-attn bank split: the "q" leg is the 2×-wide
            // [query | gate] pack (`use_fused_qgkv` in the hand-written
            // body: launch_split_qkv_bf16(packed, qg, k, v, N, 2*Hq, Hk)).
            kernels::launch_split_qkv_bf16(
                ws.gate_up_fused.data(),
                la.fa_qg_packed.data(), ws.k.data(), ws.v.data(),
                N, 2 * Hq, Hk, stream);
            break;
        }
        case PieForwardOpKind::SplitGdn: {
            // Two flavors, told apart by their traced widths: the qkvz row
            // split ([conv_dim | V_dim]) and the interleaved b/a split
            // ([V_h | V_h]) — family.rs's fused gdn body.
            if (op.param0 == static_cast<std::uint32_t>(conv_dim) &&
                op.param1 == static_cast<std::uint32_t>(V_dim)) {
                kernels::launch_split_bf16_rows(
                    la.mixed_qkvz.data(), la.mixed_qkv.data(), la.z.data(),
                    N, conv_dim, V_dim, stream);
            } else if (op.param0 == static_cast<std::uint32_t>(V_h) &&
                       op.param1 == static_cast<std::uint32_t>(V_h)) {
                kernels::launch_split_qwen_gdn_ba_bf16(
                    la.ba.data(), la.b.data(), la.a.data(), N, V_h, stream);
            } else {
                throw_drift("SplitGdn widths (" +
                            std::to_string(op.param0) + ", " +
                            std::to_string(op.param1) +
                            ") match neither the qkvz nor the ba split");
            }
            break;
        }
        case PieForwardOpKind::CausalConv1d: {
            const std::string_view name = plan.weight_name(op);
            const ParsedWeightName nm = parse_weight_name(name);
            if (nm.field != "conv") throw_unknown_weight(name);
            const auto& layer = layer_of(w, nm, name);
            if (op.param1 != static_cast<std::uint32_t>(conv_K)) {
                throw_drift("CausalConv1d kernel " +
                            std::to_string(op.param1) + " != config's " +
                            std::to_string(conv_K));
            }
            const int state_layer = static_cast<int>(op.param0);
            require(layer.la_conv1d_w, name);
            // Decode-update lowering against the layer's per-request conv
            // state — the hand-written `linear_decode` branch, verbatim
            // (batched slot-indirected kernel, or the legacy slot-0 path
            // for the parity entry point).
            if (slot_ids_d != nullptr) {
                kernels::launch_causal_conv1d_update_batched_bf16(
                    la.mixed_qkv.data(), layer.la_conv1d_w->data(),
                    layer.la_conv1d_b ? layer.la_conv1d_b->data() : nullptr,
                    state_cache.conv_state(state_layer, /*slot=*/0),
                    slot_ids_d,
                    static_cast<long long>(state_cache.conv_kernel()) *
                        state_cache.conv_dim(),
                    la.mixed_qkv_post.data(),
                    R, conv_dim, conv_K, stream);
            } else {
                kernels::launch_causal_conv1d_update_bf16(
                    la.mixed_qkv.data(), layer.la_conv1d_w->data(),
                    layer.la_conv1d_b ? layer.la_conv1d_b->data() : nullptr,
                    state_cache.conv_state(state_layer, 0),
                    la.mixed_qkv_post.data(),
                    conv_dim, conv_K, stream);
            }
            break;
        }
        case PieForwardOpKind::GdnPrep: {
            // The one kind naming TWO weights: a_log in the weight slot,
            // dt_bias as a param0 name index (pie_forward.h's op table).
            const std::string_view name = plan.weight_name(op);
            const ParsedWeightName nm = parse_weight_name(name);
            if (nm.field != "a_log") throw_unknown_weight(name);
            const std::string_view dt_name = plan.name(op.param0);
            const ParsedWeightName dt_nm = parse_weight_name(dt_name);
            if (dt_nm.field != "dt_bias" || dt_nm.layer != nm.layer) {
                throw_unknown_weight(dt_name);
            }
            const auto& layer = layer_of(w, nm, name);
            if (layer.la_A_log_fp32 == nullptr) throw_unknown_weight(name);
            kernels::launch_qwen_gdn_post_conv_prep_bf16(
                la.mixed_qkv_post.data(), la.a.data(), la.b.data(),
                layer.la_A_log_fp32,
                require(layer.la_dt_bias, dt_name)->data(),
                la.q_pre.data(), la.k_pre.data(), la.v_fp32.data(),
                la.g_log.data(), la.beta.data(),
                N, K_h, V_h, K_d, V_d, conv_dim, stream);
            // GQA materialisation is a LOWERING of the recurrence, not a
            // trace op: the decode GQA step kernel indexes the compact
            // K_h-head layout directly, so repeat_interleave launches only
            // when that kernel is not eligible — the hand-written decode
            // predicate (`V_h != K_h && !use_decode_gqa_recurrent`; the
            // warp-tiled / batched-FLA terms are prefill-only and false on
            // this slice).
            if (V_h != K_h && !use_decode_gqa_recurrent) {
                kernels::launch_repeat_interleave_heads_fp32(
                    la.q_pre.data(), la.q_norm.data(), N, K_h, V_h, K_d,
                    stream);
                kernels::launch_repeat_interleave_heads_fp32(
                    la.k_pre.data(), la.k_norm.data(), N, K_h, V_h, K_d,
                    stream);
            }
            break;
        }
        case PieForwardOpKind::GatedDelta: {
            // The decode recurrent step against the layer's per-request
            // recurrent slab — the hand-written `linear_decode` recurrence
            // branch, kernel for kernel (batched GQA / batched / legacy
            // single-request, each in its state dtype).
            const int state_layer = static_cast<int>(op.param0);
            void* state_slot0 =
                state_cache.recurrent_state_raw(state_layer, /*slot=*/0);
            if (slot_ids_d != nullptr) {
                if (use_decode_gqa_recurrent) {
                    if (state_bf16) {
                        kernels::launch_recurrent_gated_delta_step_batched_gqa_state_bf16(
                            la.q_pre.data(), la.k_pre.data(),
                            la.v_fp32.data(), la.g_log.data(),
                            la.beta.data(),
                            state_slot0, slot_ids_d, slot_stride,
                            la.core_out.data(),
                            R, K_h, V_h, K_d, V_d, stream);
                    } else {
                        kernels::launch_recurrent_gated_delta_step_batched_gqa(
                            la.q_pre.data(), la.k_pre.data(),
                            la.v_fp32.data(), la.g_log.data(),
                            la.beta.data(),
                            static_cast<float*>(state_slot0),
                            slot_ids_d, slot_stride,
                            la.core_out.data(),
                            R, K_h, V_h, K_d, V_d, stream);
                    }
                } else {
                    if (state_bf16) {
                        kernels::launch_recurrent_gated_delta_step_batched_state_bf16(
                            q_recur_full, k_recur_full,
                            la.v_fp32.data(), la.g_log.data(),
                            la.beta.data(),
                            state_slot0, slot_ids_d, slot_stride,
                            la.core_out.data(),
                            R, V_h, K_d, V_d, stream);
                    } else {
                        kernels::launch_recurrent_gated_delta_step_batched(
                            q_recur_full, k_recur_full,
                            la.v_fp32.data(), la.g_log.data(),
                            la.beta.data(),
                            static_cast<float*>(state_slot0),
                            slot_ids_d, slot_stride,
                            la.core_out.data(),
                            R, V_h, K_d, V_d, stream);
                    }
                }
            } else {
                if (state_bf16) {
                    kernels::launch_recurrent_gated_delta_step_state_bf16(
                        q_recur_full, k_recur_full,
                        la.v_fp32.data(), la.g_log.data(), la.beta.data(),
                        state_slot0, la.core_out.data(),
                        /*B=*/1, V_h, K_d, V_d, stream);
                } else {
                    kernels::launch_recurrent_gated_delta_step(
                        q_recur_full, k_recur_full,
                        la.v_fp32.data(), la.g_log.data(), la.beta.data(),
                        static_cast<float*>(state_slot0),
                        la.core_out.data(),
                        /*B=*/1, V_h, K_d, V_d, stream);
                }
            }
            break;
        }
        case PieForwardOpKind::RmsnormGated: {
            // core_out (fp32) → fused z-gated RMSNorm → bf16, per (n, h)
            // row of V_d — the hand-written fused kernel, one launch.
            const std::string_view name = plan.weight_name(op);
            const ParsedWeightName nm = parse_weight_name(name);
            if (nm.field != "gate_norm") throw_unknown_weight(name);
            const auto& layer = layer_of(w, nm, name);
            if (layer.la_norm_w_fp32 == nullptr) throw_unknown_weight(name);
            kernels::launch_rmsnorm_gated_fp32_in_bf16(
                la.core_out.data(), la.z.data(), layer.la_norm_w_fp32,
                la.core_out_bf16.data(),
                N * V_h, V_d, /*eps=*/eps, stream);
            break;
        }
        case PieForwardOpKind::SplitQGate: {
            // Interleaved per-head [query | gate] de-interleave of the
            // 2×-wide q pack.
            if (op.param0 != static_cast<std::uint32_t>(num_q_heads) ||
                op.param1 != static_cast<std::uint32_t>(d)) {
                throw_drift("SplitQGate geometry (" +
                            std::to_string(op.param0) + ", " +
                            std::to_string(op.param1) +
                            ") != config's heads/head_dim");
            }
            kernels::launch_split_q_gate_bf16(
                la.fa_qg_packed.data(), ws.q.data(), la.fa_gate.data(),
                N, num_q_heads, d, stream);
            break;
        }
        case PieForwardOpKind::RmsnormPerHead: {
            // Gemma fold, in place, one row per head — the hand-written
            // q/k norms (`launch_rmsnorm_gemma_bf16` over N·heads rows).
            if (op.param1 !=
                static_cast<std::uint32_t>(PieForwardNormVariant::Gemma)) {
                throw_drift("only the Gemma per-head norm is emitted");
            }
            const std::string_view name = plan.weight_name(op);
            const ParsedWeightName nm = parse_weight_name(name);
            const auto& layer = layer_of(w, nm, name);
            if (nm.field == "q_norm") {
                kernels::launch_rmsnorm_gemma_bf16(
                    ws.q.data(), require(layer.fa_q_norm, name)->data(),
                    ws.q.data(), N * num_q_heads, d, eps, stream);
            } else if (nm.field == "k_norm") {
                kernels::launch_rmsnorm_gemma_bf16(
                    ws.k.data(), require(layer.fa_k_norm, name)->data(),
                    ws.k.data(), N * num_kv_heads, d, eps, stream);
            } else {
                throw_unknown_weight(name);
            }
            break;
        }
        case PieForwardOpKind::Rope: {
            // Partial rope: param1 is the resolved rotary channel count
            // (validated against the driver's own derivation at build).
            if (op.param0 !=
                    static_cast<std::uint32_t>(PieForwardRopeKind::Standard) ||
                op.param1 == 0) {
                throw_drift("only the partial standard rope is emitted");
            }
            kernels::launch_rope_partial_bf16(
                ws.q.data(), ws.k.data(), positions,
                N, num_q_heads, num_kv_heads,
                d, static_cast<int>(op.param1), cfg.rope_theta, stream);
            break;
        }
        case PieForwardOpKind::KvAppend: {
            // The trace states the MODEL layer; the compact KV slot is
            // storage knowledge this emitter derives from the binding
            // (`Qwen3_5LayerWeights::kv_layer` — family.rs's
            // full_attn_body doc states exactly this division).
            const int model_layer = static_cast<int>(op.param0);
            if (model_layer < 0 ||
                model_layer >= static_cast<int>(w.layers.size()) ||
                w.layers[model_layer].kv_layer < 0) {
                throw_drift("KvAppend layer " +
                            std::to_string(model_layer) +
                            " has no KV cache slot");
            }
            auto kv_view = cache.layer_view(w.layers[model_layer].kv_layer);
            if (has_write_desc) {
                kernels::launch_write_kv_explicit_bf16(
                    kv_view, ws.k.data(), ws.v.data(),
                    w_page_d, w_off_d, N, stream, row_valid_d);
            } else {
                kernels::launch_write_kv_to_pages(
                    kv_view, ws.k.data(), ws.v.data(),
                    qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, N, R, stream);
            }
            break;
        }
        case PieForwardOpKind::Attention: {
            const int model_layer = static_cast<int>(op.param0);
            if (model_layer < 0 ||
                model_layer >= static_cast<int>(w.layers.size()) ||
                w.layers[model_layer].kv_layer < 0) {
                throw_drift("Attention layer " +
                            std::to_string(model_layer) +
                            " has no KV cache slot");
            }
            auto kv_view = cache.layer_view(w.layers[model_layer].kv_layer);
            // Plan choice is a per-fire LOWERING (family.rs's
            // full_attn_body doc): the same four-way branch as the
            // hand-written body, same inputs, same order. On a decode fire
            // the decode plan is the live branch; the others cover the
            // force_prefill_path parity mode.
            const bool use_small_prefill_naive =
                decode_plan == nullptr &&
                prefill_plan == nullptr &&
                fwd_cfg.small_prefill_naive_attention_max_tokens > 0 &&
                N <= fwd_cfg.small_prefill_naive_attention_max_tokens &&
                kv_view.is_native_bf16() && !kv_view.hnd_layout;
            if (decode_plan) {
                ops::dispatch_attention_flashinfer_decode(
                    *decode_plan,
                    ws.q.data(), kv_view, ws.attn_out.data(),
                    kv_page_indices, kv_page_indptr, kv_last_page_lens,
                    attn_ws, stream);
            } else if (prefill_plan) {
                ops::dispatch_attention_flashinfer_prefill_bf16(
                    *prefill_plan,
                    ws.q.data(), kv_view.k_bf16_pages, kv_view.v_bf16_pages,
                    ws.attn_out.data(),
                    qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, attn_ws, stream);
            } else if (use_small_prefill_naive) {
                ops::launch_attention_naive_paged_bf16(
                    ws.q.data(), kv_view.k_bf16_pages, kv_view.v_bf16_pages,
                    ws.attn_out.data(),
                    qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens,
                    N, R, num_q_heads, num_kv_heads, d,
                    cache.page_size(), stream);
            } else {
                ops::launch_attention_flashinfer_prefill(
                    ws.q.data(), kv_view, ws.attn_out.data(),
                    qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens,
                    qo_indptr_h, kv_page_indptr_h,
                    N, R, num_q_heads, attn_ws, stream);
            }
            break;
        }
        case PieForwardOpKind::SigmoidGateMul: {
            // attn_out *= sigmoid(gate) — the full-attention output gate.
            kernels::launch_sigmoid_gate_inplace_bf16(
                ws.attn_out.data(), la.fa_gate.data(), N * Hq, stream);
            break;
        }
        case PieForwardOpKind::Swiglu: {
            if (gate_up_used_fused) {
                kernels::launch_chunked_swiglu_bf16(
                    ws.gate_up_fused.data(), ws.gate.data(), N, I, stream);
            } else {
                kernels::launch_swiglu_bf16(
                    ws.gate.data(), ws.up.data(), ws.gate.data(),
                    N * I, stream);
            }
            break;
        }
        case PieForwardOpKind::LmHead: {
            const std::string_view name = plan.weight_name(op);
            // Tied embeddings trace the lm head as "embed"; either way the
            // binding already aliased `w.lm_head` accordingly.
            const DeviceTensor* lm_head =
                name == "embed" ? require(w.embed, name)
                : name == "lm_head" ? require(w.lm_head, name)
                : nullptr;
            if (lm_head == nullptr) throw_unknown_weight(name);
            // The hand-written epilogue, copied whole: the final norm
            // already landed ALL rows in norm_x (the Rmsnorm arm above);
            // compact-logit fires gather the sampler rows into norm_y and
            // multiply just those, full emits multiply everything. Then
            // the full normed hidden is copied back to ws.y for MTP/state
            // plumbing — a fire-shape service the trace does not state,
            // exactly like the gather.
            if (logit_row_indices_d != nullptr &&
                num_logit_rows > 0 &&
                num_logit_rows < N) {
                kernels::launch_gather_bf16_rows(
                    static_cast<const std::uint16_t*>(ws.norm_x.data()),
                    logit_row_indices_d,
                    static_cast<std::uint16_t*>(ws.norm_y.data()),
                    num_logit_rows, H, stream);
                ops::gemm_act_x_w(cublas.handle(),
                    ws.norm_y.data(), *lm_head,
                    ws.logits.data(), num_logit_rows, V, H);
            } else {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.norm_x.data(), *lm_head,
                    ws.logits.data(), N, V, H);
            }
            CUDA_CHECK(cudaMemcpyAsync(
                ws.y.data(), ws.norm_x.data(),
                static_cast<std::size_t>(N) * H * sizeof(std::uint16_t),
                cudaMemcpyDeviceToDevice, stream));
            break;
        }
        default:
            throw std::runtime_error(
                "declared qwen35 forward: op kind " +
                std::to_string(static_cast<std::uint32_t>(op.kind)) +
                " has no emission rule");
        }
    }
}

}  // namespace pie_cuda_driver::model
