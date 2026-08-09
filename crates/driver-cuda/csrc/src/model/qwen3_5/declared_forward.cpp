#include "attention_workspace.hpp"
#include "model/qwen3_5/declared_forward.hpp"
#include "model/qwen3_5/qwen3_5_moe.hpp"
#include "model/qwen3_5/qwen3_5_moe_forward.hpp"
#include "moe/moe_dispatch.hpp"
#include "moe/moe_grouped_gemm.hpp"
#include "moe/topk_softmax.hpp"
#include <type_traits>
#include "model/declared/arms.hpp"
#include "model/declared/execute.hpp"
#include "model/declared/registry.hpp"
#include "model/declared/weights.hpp"

#include <algorithm>
#include <charconv>
#include <atomic>

#include "model/declared/value_arena.hpp"
#include <cstdio>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "ssm/causal_conv1d.hpp"
#include "layout/deinterleave.hpp"
#include "layout/embed.hpp"
#include "ssm/gated_delta_net.hpp"
#include "layout/gather_rows.hpp"
#include "attn/kv_paged.hpp"
#include "norm/rmsnorm.hpp"
#include "rope/rope.hpp"
#include "attn/split_packed.hpp"
#include "mlp/swiglu.hpp"
#include "attn/attention_flashinfer.hpp"
#include "attn/attention_naive_paged.hpp"
#include "gemm/gemm.hpp"

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
// The name grammar is `model/declared/weights.hpp`'s (it was copied here
// byte-for-byte; that duplication is what said the executors wanted to be
// one).
using declared::ParsedWeightName;
using declared::parse_weight_name;
using declared::throw_unknown_weight;

const DeviceTensor* bind_qwen3_5_weight(
    const void* ctx, const ParsedWeightName& nm, std::string_view name);

// The MoE family's half of `declared::WeightBinder` — the same TRACE
// vocabulary the dense binder answers, over the MoE struct's own spellings,
// plus the MoE block's own names.
const DeviceTensor* bind_qwen3_5_moe_weight(
    const void* ctx, const ParsedWeightName& nm, std::string_view name)
{
    const auto& w = *static_cast<const Qwen3_5MoeWeights*>(ctx);
    if (nm.layer < 0) {
        if (nm.field == "embed") return w.embed;
        if (nm.field == "final_norm") return w.final_norm;
        if (nm.field == "lm_head") return w.lm_head;
        throw_unknown_weight(name);
    }
    if (nm.layer >= static_cast<int>(w.layers.size())) throw_unknown_weight(name);
    const Qwen3_5MoeLayerWeights& l = w.layers[static_cast<std::size_t>(nm.layer)];
    if (nm.field == "attn_norm") return l.attn_norm_pre;
    if (nm.field == "mlp_norm") return l.mlp_norm_pre;
    if (nm.field == "in_proj_qkv") return l.la_in_proj_qkv;
    if (nm.field == "in_proj_z") return l.la_in_proj_z;
    if (nm.field == "in_proj_b") return l.la_in_proj_b;
    if (nm.field == "in_proj_a") return l.la_in_proj_a;
    if (nm.field == "conv") return l.la_conv1d_w;
    if (nm.field == "conv_bias") return l.la_conv1d_b;
    if (nm.field == "dt_bias") return l.la_dt_bias;
    if (nm.field == "out_proj") return l.la_out_proj;
    if (nm.field == "q_proj") return l.fa_q_proj;
    if (nm.field == "k_proj") return l.fa_k_proj;
    if (nm.field == "v_proj") return l.fa_v_proj;
    // Same rule the dense binder states: `o_proj` is the trace's name for
    // whichever bank this LAYER KIND projects out of.
    if (nm.field == "o_proj") {
        return l.kind == Qwen3_5MoeLayerWeights::Kind::FullAttn ? l.fa_o_proj
                                                                : l.la_out_proj;
    }
    if (nm.field == "q_norm") return l.fa_q_norm;
    if (nm.field == "k_norm") return l.fa_k_norm;
    if (nm.field == "router") return l.moe_router;
    // The `{e}` is literal: the trace names the BANK, spelled the way the
    // family spells a per-expert weight, and the grouped kernel indexes it
    // by the block's expert id. There is no per-expert tensor to resolve.
    if (nm.field == "expert.{e}.gate_up") return l.moe_gate_up_proj;
    if (nm.field == "expert.{e}.down") return l.moe_down_proj;
    if (nm.field == "shared_expert.gate_up") return l.shared_gate_up_proj;
    if (nm.field == "shared_expert.down") return l.shared_down_proj;
    if (nm.field == "shared_expert_gate") return l.shared_gate_proj;
    throw_unknown_weight(name);
}

// Which binder a weights type wants. `WeightBinder` is type-erased, so this
// is the whole of what the two families need to share.
template <class W> declared::WeightBinder::Fn binder_for();
template <> declared::WeightBinder::Fn binder_for<Qwen3_5Weights>() {
    return &bind_qwen3_5_weight;
}
template <> declared::WeightBinder::Fn binder_for<Qwen3_5MoeWeights>() {
    return &bind_qwen3_5_moe_weight;
}

// This family's half of `declared::WeightBinder`. Note `attn_norm` /
// `mlp_norm`: the SAME traced names llama_like binds, spelled `_pre` in
// this weights struct — the difference an arm must never see.
const DeviceTensor* bind_qwen3_5_weight(
    const void* ctx, const ParsedWeightName& nm, std::string_view name)
{
    const auto& w = *static_cast<const Qwen3_5Weights*>(ctx);
    if (nm.layer < 0) {
        if (nm.field == "embed") return w.embed;
        if (nm.field == "final_norm") return w.final_norm;
        if (nm.field == "lm_head") return w.lm_head;
        throw_unknown_weight(name);
    }
    if (nm.layer >= static_cast<int>(w.layers.size())) {
        throw_unknown_weight(name);
    }
    const Qwen3_5LayerWeights& l =
        w.layers[static_cast<std::size_t>(nm.layer)];
    // Same TRACE vocabulary as llama_like's binder — note `attn_norm` /
    // `mlp_norm`, which this weights struct spells `_pre`. That difference
    // is exactly what an arm must never see.
    if (nm.field == "attn_norm") return l.attn_norm_pre;
    if (nm.field == "mlp_norm") return l.mlp_norm_pre;
    if (nm.field == "gate_up") return l.gate_up_proj_fused;
    if (nm.field == "gate_proj") return l.gate_proj;
    if (nm.field == "up_proj") return l.up_proj;
    if (nm.field == "down") return l.down_proj;
    // Full-attention layers.
    if (nm.field == "q_proj") return l.fa_q_proj;
    if (nm.field == "k_proj") return l.fa_k_proj;
    if (nm.field == "v_proj") return l.fa_v_proj;
    // ONE traced name, resolved by the LAYER KIND — the hybrid's two
    // attentions each land their output through the residual, and the
    // declaration says "the output projection of this layer" rather
    // than which family's bank holds it. The AOT emitter has always
    // read it this way (`emit_qwen35`'s `is_full(layer)` branch); this
    // binder did not, so every GDN layer resolved to the
    // full-attention bank and came back null. That is why the declared
    // path had never run a hybrid: layer 0 is Linear.
    if (nm.field == "o_proj") {
        return l.kind == Qwen3_5LayerWeights::Kind::FullAttn ? l.fa_o_proj
                                                             : l.la_out_proj;
    }
    if (nm.field == "q_norm") return l.fa_q_norm;
    if (nm.field == "k_norm") return l.fa_k_norm;
    if (nm.field == "qgkv") return l.fa_qgkv_proj_fused;
    // Gated-DeltaNet (linear-attention) layers. `a_log` / `gate_norm` are
    // pre-converted fp32 arrays, not tensors — the GDN arms read those off
    // the layer directly; a tensor binder has nothing to say about them.
    if (nm.field == "in_proj_qkv") return l.la_in_proj_qkv;
    if (nm.field == "in_proj_z") return l.la_in_proj_z;
    if (nm.field == "in_proj_a") return l.la_in_proj_a;
    if (nm.field == "in_proj_b") return l.la_in_proj_b;
    if (nm.field == "out_proj") return l.la_out_proj;
    if (nm.field == "dt_bias") return l.la_dt_bias;
    if (nm.field == "conv") return l.la_conv1d_w;
    throw_unknown_weight(name);
}

template <class W>
const auto& layer_of(
    const W& w, const ParsedWeightName& nm,
    std::string_view name)
{
    if (nm.layer < 0 || nm.layer >= static_cast<int>(w.layers.size())) {
        throw_unknown_weight(name);
    }
    return w.layers[nm.layer];
}

// The DENSE view, and a refusal where `make_weight_view` used to be.
// llama_like's `dense` verbatim, and for its reason: a semantic
// `Matmul` means a weight read directly, so a layer that carries a
// quant descriptor while the statement records one is facts drift.
WeightView dense(const DeviceTensor& t,
                 const std::optional<QuantMeta>& meta,
                 std::string_view name) {
    if (meta.has_value()) {
        throw std::runtime_error(
            "declared forward: '" + std::string(name) +
            "' is stored quantized but the trace records a dense Matmul "
            "over it -- the facts this class was traced with say the "
            "deployment is bf16 (MatW::repr)");
    }
    return WeightView(t);
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

// Rung 4c-iii: the launcher registry — every kernel a qwen3_5 class
// trace may STATE (dsl::cuda's raw signatures), one enum value per
// launcher symbol. The executor's Launch arm resolves and BINDS; a
// symbol outside this vocabulary means the trace and this executor
// drifted, and `qwen35_validate_stated_kernels` makes that a model-load
// failure.


// Rung 3, second family: the static C++ form of the decode/prefill
// class traces, emitted by `cargo run -p pie-forward --bin emit-cuda`
// and committed. Digest-gated like the llama forms.
#include "model/qwen3_5/generated/qwen3_5_0_8b.inc"

bool q35_generated_forward_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_DECLARED_FORWARD_GENERATED");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

}  // namespace

void qwen35_validate_stated_kernels(const pie_forward::ForwardPlan& plan) {
    const std::size_t n = plan.op_count();
    for (std::size_t i = 0; i < n; ++i) {
        const pie_forward::PieForwardOp& op = plan.op(i);
        if (op.kind == pie_forward::PieForwardOpKind::Launch) {
            (void)declared::resolve_kernel(plan.weight_name(op));
        }
    }
}

bool qwen35_declared_moe_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_DECLARED_MOE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

// `PIE_DECLARED_HOST_ARENA=0` puts this family's pin table back in
// charge; the host assigns otherwise. The A/B the conversion is checked
// against is `cuda_declared_family_parity`'s `qwen3_5_dense` row, run
// with and without it.
//
// `_LO`/`_HI` window the host's half by OWNER id — gemma-4's bisect cut,
// and the reasoning for that axis is written out there.
bool qwen35_host_arena_enabled() {
    const char* v = std::getenv("PIE_DECLARED_HOST_ARENA");
    return v == nullptr || v[0] != '0';
}

std::size_t host_arena_lo() {
    const char* v = std::getenv("PIE_DECLARED_HOST_ARENA_LO");
    return v != nullptr ? static_cast<std::size_t>(std::atoll(v)) : 0;
}

std::size_t host_arena_hi() {
    const char* v = std::getenv("PIE_DECLARED_HOST_ARENA_HI");
    return v != nullptr ? static_cast<std::size_t>(std::atoll(v))
                        : static_cast<std::size_t>(-1);
}

bool qwen35_declared_exec_trace_enabled() {
    static const bool enabled =
        std::getenv("PIE_DECLARED_FORWARD_TRACE") != nullptr;
    return enabled;
}

// The executor, over EITHER weights family.
//
// A template rather than accessors: every `layer.<field>` below is checked
// against the struct the fire actually has, so a field the MoE struct spells
// differently is a compile error rather than a silent read of the wrong
// tensor. The two families share no base -- this stands in for one.
template <class W>
bool forward_declared_tmpl(
    const Qwen35DeclaredPlan& declared,
    const W& w,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    const Qwen3_5PlanState& plan_state,
    Workspace& ws,
    Qwen3_5MoeMlpWorkspace* moe_ws,
    Qwen3_5LinearAttnWorkspace& la,
    KvCache& cache,
    RecurrentStateCache& state_cache,
    AttentionWorkspace& attn_ws,
    kernels::gemm::CublasHandle& cublas,
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
    int num_logit_rows,
    const std::int32_t* commit_lens,
    const StageHooks* stage_hooks)
{
    // Weights reach the arms only through the binder (see its header).
    using LayerW = std::decay_t<decltype(w.layers[0])>;
    constexpr bool kIsDense = std::is_same_v<W, Qwen3_5Weights>;
    const declared::WeightBinder wb{binder_for<W>(), &w};
    // Rung 4c-iii: normal decode/prefill fires walk the CLASS trace, in
    // which the declaration stated every kernel; the MTP/verify/legacy
    // service fires keep the semantic walk until 4c-iv brings their
    // classes. The state-dtype term is the build-time default's per-fire
    // cross-check (declared_facts.hpp) — a mismatch falls back, loudly.
    const bool commit_advance = commit_lens != nullptr;
    const bool state_only = num_logit_rows < 0;
    const bool state_dtype_ok =
        state_cache.recurrent_state_bf16() == declared.cuda_state_bf16;
    // 4c-iv: the service classes route too. Frozen-verify fires stay
    // semantic (their class — the stash-writing prefill — is the next
    // slice), and a commit fire whose live stash disagrees with the
    // traced fact falls back rather than replaying from a stash that is
    // not there.
    const bool commit_stash_ok =
        state_cache.verify_hidden_stash_enabled() == declared.cuda_verify_stash;
    const pie_forward::ForwardPlan* class_plan = nullptr;
    if (state_dtype_ok && slot_ids_d != nullptr &&
        (is_pure_decode || qo_indptr != nullptr)) {
        const bool frozen = state_cache.verify_frozen();
        if (frozen && !commit_advance && !state_only) {
            // The frozen-verify class: stash stores are stated iff the
            // traced fact says so; a live/fact disagreement falls back.
            if (declared.frozen_verify && commit_stash_ok) {
                class_plan = &declared.frozen_verify;
            }
        } else if (!frozen && commit_advance && !state_only) {
            if (declared.commit_advance && commit_stash_ok) {
                class_plan = &declared.commit_advance;
            }
        } else if (!frozen && state_only && !commit_advance) {
            if (declared.state_only) class_plan = &declared.state_only;
        } else if (!frozen && !commit_advance && !state_only &&
                   declared.decode && declared.prefill) {
            class_plan =
                is_pure_decode ? &declared.decode : &declared.prefill;
        }
    }
    // RUNG 5: the semantic walk is DELETED from this executor. Every
    // batched fire has a class; anything without one (legacy slot-less
    // harness fires, live-fact mismatches) falls back to the hand-written
    // path — the caller runs it when we return false.
    if (class_plan == nullptr) {
        if (qwen35_declared_exec_trace_enabled()) {
            std::fprintf(stderr,
                         "[declared-qwen35-exec] no class for fire "
                         "N=%d R=%d decode=%d commit=%d state_only=%d "
                         "frozen=%d -> hand-written\n",
                         total_tokens, num_requests, is_pure_decode ? 1 : 0,
                         commit_advance ? 1 : 0, state_only ? 1 : 0,
                         state_cache.verify_frozen() ? 1 : 0);
        }
        return false;
    }
    if (!state_dtype_ok) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::fprintf(stderr,
                         "[declared-qwen35-exec] recurrent-state dtype "
                         "differs from the build-time default; class "
                         "traces disabled, semantic walk serves\n");
        }
    }
    // The static form (decode/prefill classes; the services stay on the
    // interpreter walk). Digest-gated: a mismatch prints once under the
    // trace env and the interpreter serves, loudly recoverable.
    // The generated bodies are emitted per DENSE deployment digest and take
    // the dense weights by reference, so only that instantiation has them.
    if constexpr (kIsDense) {
    if (state_dtype_ok && q35_generated_forward_enabled()) {
        if (declared.facts_digest == kQ35GeneratedDigest_qwen3_5_0_8b) {
            // EVERY class emits (rung 3, second family, full width).
            const auto run = [&](auto fn) {
                fn(w, cfg, fwd_cfg, plan_state, ws, la, cache, state_cache,
                   attn_ws, cublas,
                   token_ids, positions, qo_indptr,
                   kv_page_indices, kv_page_indptr, kv_last_page_lens,
                   qo_indptr_h, kv_page_indptr_h,
                   total_tokens, num_requests,
                   w_page_d, w_off_d, row_valid_d, has_write_desc,
                   slot_ids_h, is_fresh_h, slot_ids_d, is_fresh_d,
                   logit_row_indices_d, num_logit_rows,
                   stage_hooks);
            };
            if (class_plan == &declared.decode) {
                run(generated_qwen35_decode_qwen3_5_0_8b);
                return true;
            }
            if (class_plan == &declared.prefill) {
                run(generated_qwen35_prefill_qwen3_5_0_8b);
                return true;
            }
            if (class_plan == &declared.state_only) {
                run(generated_qwen35_state_only_qwen3_5_0_8b);
                return true;
            }
            if (class_plan == &declared.frozen_verify) {
                run(generated_qwen35_frozen_verify_qwen3_5_0_8b);
                return true;
            }
            if (class_plan == &declared.commit_advance) {
                generated_qwen35_commit_advance_qwen3_5_0_8b(
                    w, cfg, fwd_cfg, plan_state, ws, la, cache, state_cache,
                    attn_ws, cublas,
                    token_ids, positions, qo_indptr,
                    kv_page_indices, kv_page_indptr, kv_last_page_lens,
                    qo_indptr_h, kv_page_indptr_h,
                    total_tokens, num_requests,
                    w_page_d, w_off_d, row_valid_d, has_write_desc,
                    slot_ids_h, is_fresh_h, slot_ids_d, is_fresh_d,
                    logit_row_indices_d, num_logit_rows,
                    stage_hooks, commit_lens);
                return true;
            }
        } else if (qwen35_declared_exec_trace_enabled()) {
            std::fprintf(stderr,
                         "[declared-qwen35-generated] digest mismatch:\n"
                         "  live:    %s\n  emitted: %s\n",
                         declared.facts_digest.c_str(),
                         kQ35GeneratedDigest_qwen3_5_0_8b);
        }
    }
    }  // if constexpr (dense): the generated bodies
    const pie_forward::ForwardPlan& plan = *class_plan;
    // Say ONCE, unconditionally, that this drive took a fire.
    //
    // Every other declared executor does; this one only said so under
    // `PIE_DECLARED_FORWARD_TRACE`, which is not good enough for the
    // reason `cuda_declared_family_parity` states out loud: a declared
    // side that silently DECLINES produces a record identical to the
    // hand-written side by construction, so the gate passes while
    // proving nothing. That harness refuses a run it cannot hear, and
    // it could not hear this family at all.
    {
        static std::atomic<bool> said[2] = {{false}, {false}};
        if (!said[is_pure_decode ? 0 : 1].exchange(true)) {
            std::fprintf(stderr,
                         "[declared-qwen35] first %s fire: N=%d R=%d "
                         "ops=%zu\n",
                         is_pure_decode ? "DECODE" : "PREFILL",
                         total_tokens, num_requests, plan.op_count());
        }
    }
    if (qwen35_declared_exec_trace_enabled()) {
        std::fprintf(stderr,
                     "[declared-qwen35-exec] N=%d R=%d decode=%d ops=%zu "
                     "class=1\n",
                     total_tokens, num_requests, is_pure_decode ? 1 : 0,
                     plan.op_count());
    }
    // Both fire shapes run here now (arc 3): the trace is decode/prefill-
    // agnostic by design (crates/model-compiler/src/trace.rs — CausalConv1d / GatedDelta /
    // Attention are opaque state ops whose lowering the emitter picks per
    // fire), so the state-op arms below branch on `is_pure_decode` exactly
    // as the hand-written `linear_attn_layer_body` branches. A MIXED fire
    // (prefill + decode rows co-batched) is not separate machinery: the
    // hand-written body treats any `is_pure_decode == false` fire as one
    // qo_indptr-windowed prefill shape (a decode row is just an Nr == 1
    // window), and the walk mirrors that single shape.

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
                return layer.kind == LayerW::Kind::FullAttn;
            });
        if (w_page_d == nullptr || w_off_d == nullptr ||
            !cache.format().is_native_bf16() || !has_full_attention) {
            throw std::runtime_error(
                "Qwen3.5 explicit KV writes are unsupported by this layout");
        }
    }

    // MTP-adjacent fire shapes (this arc). Both are per-fire SERVICES
    // around the one traced pass (family.rs's epilogue doc states exactly
    // this division), so neither changes which ops the plan carries — they
    // change which arms the walk runs, mirroring where the hand-written
    // body returns early / branches:
    //  * commit-advance (`commit_lens != nullptr`): the spec-decode repair
    //    re-runs ONLY each linear layer's conv+prep+recurrence over the
    //    confirmed prefix, loading the layer's in-proj activations from the
    //    verify stash (rs_buffer_fold is gate-excluded, so the stash is the
    //    only source). No embed/norms/attention/MLP/epilogue.
    //  * state-only (`num_logit_rows < 0`): the speculative repair's
    //    whole-backbone flavor — everything runs except the final-norm /
    //    lm_head epilogue (the hand-written `if (num_logit_rows < 0 ||
    //    commit_advance) return;`).
    // (`commit_advance` / `state_only` hoisted above for the class-walk
    // selection.)

    // Per-slot reset for freshly (re)assigned rs slots — the hand-written
    // reset stage minus the rs-buffer branches the caller's gate excluded.
    // Commit-advance skips the reset whole: it advances the existing
    // committed state (the hand-written `commit_advance && !rs_buffer_fold`
    // arm). (Freshness occurs on a context's first fire,
    // a prefill; on a pure-decode fire the runtime guarantees no slot is
    // fresh, but the hand-written body still runs the check on both shapes,
    // so the walk runs it too rather than reasoning it away.)
    if (commit_advance) {
        // No reset: advancing the existing committed state.
    } else if (slot_ids_h != nullptr && is_fresh_h != nullptr) {
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
        // Legacy null-slot prefill: reset all (the parity entry point's
        // "fresh state before consumption" semantic, max_slots == 1).
        state_cache.reset(stream);
    }

    // Attention plan pointers, read exactly as qwen3_5_forward_paged reads
    // them (prepare hoisted the host-side planning out of the body).
    const kernels::attn::DecodePlanCache* decode_plan =
        plan_state.decode_plan ? plan_state.decode_plan.get() : nullptr;
    const kernels::attn::PrefillPlanCache* prefill_plan =
        (plan_state.use_prefill_plan && plan_state.prefill_plan)
            ? plan_state.prefill_plan.get()
            : nullptr;

    // GDN recurrent-state facts, hoisted once (constant across layers).
    const bool state_bf16 = state_cache.recurrent_state_bf16();
    const auto slot_stride = static_cast<long long>(
        state_cache.recurrent_slot_stride_floats());
    // The hand-written body's routing booleans, term for term. One of its
    // terms is a constant on this slice, resolved by the caller's gate:
    // `linear_decode = is_pure_decode && !rs_buffer_write` (rs-buffer fires
    // excluded by the Stage-2 verdict). `write_state = !verify_frozen &&
    // !rs_buffer_write`: frozen-verify fires run here with
    // write_state=false — the state-suppressing verify pass whose in-proj
    // activations the stash-write below captures for the later replay.
    const bool write_state = !state_cache.verify_frozen();

    // Verify-stash facts (`linear_attn_layer_body`'s stash block, hoisted:
    // `verify_hidden_stash_layer` is non-null for every layer exactly when
    // the stash is configured, so the per-layer null checks collapse to
    // one enabled bit). Layout per linear layer, bf16, max_tokens stride:
    //   [ mixed_qkv (conv_dim) | a (V_h) | b (V_h) ]
    // replay_load (commit-advance): load them and SKIP the in-proj GEMMs
    // and splits entirely. stash_write (frozen verify): cache them after
    // the in-proj GEMMs/splits, before the conv — same launch position.
    const bool stash_enabled = state_cache.verify_hidden_stash_enabled();
    const std::size_t stash_stride =
        static_cast<std::size_t>(state_cache.verify_stash_max_tokens());
    const std::size_t stash_a_off = stash_stride * conv_dim;
    const std::size_t stash_b_off =
        stash_a_off + stash_stride * static_cast<std::size_t>(V_h);
    auto slot_for = [&](int r) -> int {
        return slot_ids_h ? slot_ids_h[r] : 0;
    };
    // Decode GQA step: indexes the compact K_h-head layout directly.
    // Prefill recurrence family — the hand-written selection, verbatim:
    // warp-tiled for small-N slotted prefill (STOPGAP: only when it need
    // not persist state, unless the env re-enables the persisting fold;
    // never on commit-advance — the FLA path is the only one threading
    // commit_len); else the env-gated cached kernel; else the batched
    // GQA-aware FLA (the c>=64 spec path). `use_batched_fla_gqa` also
    // decides whether GdnPrep skips the repeat_interleave materialisation.
    //
    // What the recurrence consumes when the GQA kernels don't index the
    // compact K_h layout directly used to be a pair of host
    // indirections here (`q_recur_full`, `k_recur_full`), forking on
    // `V_h == K_h` to pick between the pre-repeat and post-repeat
    // workspace fields. Both are gone: the repeat declares its result,
    // so the cached arm states it as the recurrence's operand and the
    // fork the driver was making is the DECLARATION's.

    // Commit-advance op filter — the walk's mirror of the hand-written
    // layer loop's `if (commit_advance) { if (!is_linear) continue; ...
    // }` plus `linear_attn_layer_body`'s replay_load / `if (commit_len !=
    // nullptr) return;` skips: only conv+prep+recurrence run, preceded by
    // the in-proj GEMMs+splits ONLY when there is no stash to replay from
    // (the hand-written `replay_load` false branch — same launches, same
    // degenerate reliance on whatever norm_x holds).

    // The row axes this family does NOT state are what makes the drive
    // short. No peel (its hooks are observation-only and fire-wide), no
    // spatial mask split, no depth bands, no lora lanes — so every
    // rectangle is the whole fire, and the arms keep reading `N`. If any
    // of those axes is ever declared here, this is where the rectangle's
    // row count has to start reaching the arms, exactly as llama_like's
    // does.
    std::vector<pie_forward::PieForwardRow> rows(
        static_cast<std::size_t>(N));
    for (int r = 0; r < N; ++r) {
        pie_forward::PieForwardRow& row = rows[static_cast<std::size_t>(r)];
        row.multi_token = is_pure_decode ? 0 : 1;
        row.custom_mask = 0;
        row.hooked = stage_hooks != nullptr ? 1 : 0;
        row.lora = 0;
        row.write_desc = has_write_desc ? 1 : 0;
        row.wants_scores =
            (stage_hooks != nullptr && stage_hooks->wants_attn_score) ? 1 : 0;
        // Which rows the epilogue reads. A compact-logit fire samples a
        // subset; anything else samples every row.
        row.samples =
            (logit_row_indices_d != nullptr && num_logit_rows > 0 &&
             num_logit_rows < N)
                ? 0
                : 1;
        row._pad = 0;
        row.depth_k = -1;
    }
    if (logit_row_indices_d != nullptr && num_logit_rows > 0 &&
        num_logit_rows < N) {
        // The sampled set is a COUNT here, not a membership test: the
        // gather reads `logit_row_indices_d` itself, and the lowering
        // only needs to know how many rows the epilogue covers.
        for (int r = 0; r < num_logit_rows; ++r) {
            rows[static_cast<std::size_t>(r)].samples = 1;
        }
    }
    const pie_forward::PieForwardLowered flat =
        plan.lower(rows.data(), rows.size());
    if (flat.uncovered != pie_forward::PieForwardUncovered::None) {
        throw std::runtime_error(
            "declared qwen35 forward: the lowering refuses this fire, "
            "reason " +
            std::to_string(static_cast<std::uint32_t>(flat.uncovered)));
    }

    // THE HOST ASSIGNS, per island. `PIE_DECLARED_HOST_ARENA=0` puts the
    // pin table below back in charge, which is the A/B this family's
    // conversion is checked against (see `cuda_declared_family_parity`'s
    // `qwen3_5_dense` row).
    //
    // ONE STATEMENT THIS FAMILY CANNOT PLACE YET, recorded here because
    // the conversion has to stop at it rather than around it. The dense
    // MLP states a single `gate_up` matmul, and the driver materialises
    // it as ONE buffer or TWO depending on the binding
    // (`gate_up_proj_fused`, `arm_swiglu`'s fork). Fused, the traced
    // value is `[N, 2I]` and has a home. Unfused, the two GEMMs write
    // `ws.gate` and `ws.up`, which is `[gate rows | up rows]` and not
    // the row-interleaved `[N, 2I]` the value names -- so that
    // deployment's traced value has no single home and the arena cannot
    // give it one.
    //
    // The fix is a DECLARATION fix of the same kind gpt-oss's
    // `residual_add` order turned out to be: an unfused binding should
    // state two matmuls, because that is what it does. It is not made
    // here blind -- the facts already carry `cuda.gate_up_fused`, and it
    // wants a deployment that actually takes the unfused branch to check
    // against.
    declared::ValueArena values;
    values.reset_pins_only(plan.value_count());
    values.bind_offsets(ws.declared_values.data(), ws.declared_values.nbytes(),
                        flat);
    declared::trace_arena("qwen35", plan, flat, ws.declared_values.nbytes(),
                          N, R);

    // WHAT THE CONVENTION WAS, for every value a CONVERTED arm touches.
    //
    // A pin WINS over the host's offset, which is the migration rule
    // rather than a conflict: an arm that has not moved still writes
    // `ws.norm_x` by convention, so its consumers have to read those
    // bytes and not the ones the lowering set aside. An entry goes away
    // when its island moves, and the value falls through to the arena.
    //
    // Which entries are LIVE is decided by `movable` below, not by this
    // switch: an entry here is what the convention was, and the arena
    // takes over a value only once every op touching it reads its
    // operands off the plan.
    //
    // WHICH VALUES MAY MOVE, and why it is computed rather than listed.
    // A value can take the host's address only when EVERY op that
    // touches it has been converted -- one unconverted reader still
    // looks at a workspace field, and one unconverted writer still fills
    // one. Listing the movable set by hand is the same bookkeeping the
    // pin table already is, kept in a second place and able to drift, so
    // it is derived from the one fact that changes per island: which op
    // KINDS read their operands off the plan.
    //
    // Aliases move together or not at all. A value in an alias set whose
    // other members are still pinned would be given an address its own
    // chain does not share, which is the failure the owner table exists
    // to prevent.
    std::vector<std::uint8_t> movable(plan.value_count(), 1);
    {
        // A `Launch` is converted or not PER SYMBOL, so the predicate
        // takes the op rather than the kind. Getting this wrong in the
        // permissive direction is the failure mode: a value would take
        // an arena address while an arm still wrote a workspace field.
        const auto converted_launch = [&](const PieForwardOp& op) {
            switch (declared::resolve_kernel(plan.weight_name(op))) {
            case declared::Kernel::ConvUpdateBatched:
            case declared::Kernel::ConvPrefillBatched:
            case declared::Kernel::WriteKvExplicit:
            case declared::Kernel::WriteKvToPages:
                return true;
            case declared::Kernel::ChunkedSwiglu:
                // The DENSE caller is converted; the routed and shared
                // ones still name the MoE workspace.
                return kIsDense;
            case declared::Kernel::AttnFlashinferDecode:
            case declared::Kernel::AttnFlashinferPrefill:
                // BOTH ENDS now. The dispatches declare an output in
                // both classes -- the goldens say so, correcting an
                // earlier reading here -- so the query comes off the
                // plan and the result lands in the value, with the
                // guard's binding and `ws.attn_out` as fallbacks that
                // this deployment does not take.
                return true;
            case declared::Kernel::StepBatched:
            case declared::Kernel::StepBatchedBf16:
            case declared::Kernel::StepBatchedGqa:
            case declared::Kernel::StepBatchedGqaBf16:
            case declared::Kernel::PrefillWarpTiledGqa:
            case declared::Kernel::PrefillWarpTiledGqaBf16:
            case declared::Kernel::PrefillCached:
            case declared::Kernel::PrefillCachedBf16:
            case declared::Kernel::PrefillFla:
            case declared::Kernel::PrefillFlaBf16:
            case declared::Kernel::RepeatInterleave:
                // Both ends now. The recurrence writes the value it is
                // asked for -- its own where the decode step declares
                // one, its guard's otherwise -- and reads its five
                // operands off the plan. Under GQA q and k come from the
                // repeat instead, whose own SOURCE is stated, so the
                // chain from `q_pre` through the repeat to the
                // recurrence is consistent whichever way the deployment
                // goes: the only buffer read by convention is the
                // repeat's destination, and no traced value lives there.
                return true;
            default:
                // Notably `Swiglu`, the PAIR spelling: it reads
                // `ws.gate` and `ws.up`, which the single traced
                // `gate_up` value does not describe, so that value must
                // not move.
                return false;
            }
        };
        const auto converted = [](PieForwardOpKind k) {
            switch (k) {
            case PieForwardOpKind::Embed:
            case PieForwardOpKind::Rmsnorm:
            case PieForwardOpKind::Matmul:
            case PieForwardOpKind::SplitQkv:
            case PieForwardOpKind::SplitGdn:
            case PieForwardOpKind::GdnPrep:
            case PieForwardOpKind::RmsnormGated:
            case PieForwardOpKind::SplitQGate:
            case PieForwardOpKind::RmsnormPerHead:
            case PieForwardOpKind::SigmoidGateMul:
            case PieForwardOpKind::LmHead:
            case PieForwardOpKind::HookSite:
                return true;
            default:
                // `Guard` is deliberately NOT here, and it looks like it
                // should be: a guard writes nothing itself, its regions
                // do, and they answer for themselves. But a guard's
                // result may be written by an arm that still uses a
                // convention -- the attention's does -- and the guard is
                // the only op that names that value, so counting it
                // converted lets the value move out from under an arm
                // that has not. Tried, and the gate said so.
                return false;
            }
        };
        for (std::size_t i = 0; i < plan.op_count(); ++i) {
            const PieForwardOp& op = plan.op(i);
            if (op.kind == PieForwardOpKind::Launch ? converted_launch(op)
                                                    : converted(op.kind)) {
                continue;
            }
            for (const std::uint32_t v : plan.inputs(op)) {
                if (v < movable.size()) movable[v] = 0;
            }
            for (const std::uint32_t v : plan.outputs(op)) {
                if (v < movable.size()) movable[v] = 0;
            }
        }
        // Fold onto the alias owner, both ways: a pinned member pins the
        // set, and a member with no offset of its own follows its owner.
        // A value with no owner entry is its OWN owner, which is the
        // same rule `slot` uses. Skipping the fold when the table looks
        // short was the wrong failure direction and cost a bisect: the
        // gated attention output kept a movable flag its pinned chain
        // partner did not, so the in-place gate read `ws.attn_out` and
        // wrote the arena. An alias set moves together or not at all,
        // and a table this cannot read means NOT.
        const auto owner_of = [&](std::size_t v) -> std::size_t {
            return v < flat.value_owners_len
                       ? static_cast<std::size_t>(flat.value_owners[v])
                       : v;
        };
        std::vector<std::uint8_t> owner_ok(movable.size(), 1);
        for (std::size_t v = 0; v < movable.size(); ++v) {
            const std::size_t o = owner_of(v);
            if (!movable[v] && o < owner_ok.size()) owner_ok[o] = 0;
        }
        for (std::size_t v = 0; v < movable.size(); ++v) {
            const std::size_t o = owner_of(v);
            movable[v] = (o < owner_ok.size()) ? owner_ok[o] : 0;
        }
    }

    if (std::getenv("PIE_Q35_MOVABLE_DUMP") != nullptr) {
        for (std::size_t v = 60; v < 70 && v < movable.size(); ++v) {
            std::fprintf(stderr,
                         "[q35-movable] v%zu movable=%d owner=%u off=%zu\n",
                         v, static_cast<int>(movable[v]),
                         v < flat.value_owners_len ? flat.value_owners[v] : 0u,
                         v < flat.value_offsets_len ? flat.value_offsets[v]
                                                    : 0u);
        }
    }

    {
        const bool host_arena = qwen35_host_arena_enabled();
        const std::size_t arena_lo = host_arena_lo();
        const std::size_t arena_hi = host_arena_hi();
        const std::size_t op_count = plan.op_count();
        for (std::size_t i = 0; i < op_count; ++i) {
            const PieForwardOp& op = plan.op(i);
            const auto outs = plan.outputs(op);
            if (outs.size == 0) continue;
            const auto place = [&](std::size_t which, void* ptr) {
                if (which >= outs.size || ptr == nullptr) return;
                const std::uint32_t v = outs[which];
                if (host_arena && v < movable.size() && movable[v] != 0 &&
                    v < flat.value_offsets_len &&
                    flat.value_offsets[v] != declared::ValueArena::kNamed) {
                    const std::size_t owner =
                        v < flat.value_owners_len
                            ? static_cast<std::size_t>(flat.value_owners[v])
                            : static_cast<std::size_t>(v);
                    if (owner >= arena_lo && owner < arena_hi) return;
                }
                values.pin(v, ptr);
            };
            switch (op.kind) {
            case PieForwardOpKind::Embed:
                place(0, ws.y.data());
                break;
            case PieForwardOpKind::Rmsnorm:
                // All three sites -- attn_norm, mlp_norm, final_norm --
                // land in `norm_x`. qwen3_5's post-attention norm reads
                // it where llama_like reads `norm_y`.
                place(0, ws.norm_x.data());
                break;
            case PieForwardOpKind::Matmul: {
                const ParsedWeightName nm =
                    parse_weight_name(plan.weight_name(op));
                if (nm.field == "in_proj_qkv")      place(0, la.mixed_qkv.data());
                else if (nm.field == "in_proj_z")   place(0, la.z.data());
                else if (nm.field == "in_proj_a")   place(0, la.a.data());
                else if (nm.field == "in_proj_b")   place(0, la.b.data());
                else if (nm.field == "qgkv")        place(0, ws.gate_up_fused.data());
                else if (nm.field == "q_proj")      place(0, la.fa_qg_packed.data());
                else if (nm.field == "k_proj")      place(0, ws.k.data());
                else if (nm.field == "v_proj")      place(0, ws.v.data());
                else if (nm.field == "o_proj")      place(0, ws.y.data());
                else if (nm.field == "gate_up")     place(0, ws.gate_up_fused.data());
                // The unfused pair (2d): each half is a value of its
                // own now, and lands where the activation reads it.
                else if (nm.field == "gate_proj")   place(0, ws.gate.data());
                else if (nm.field == "up_proj")     place(0, ws.up.data());
                else if (nm.field == "down")        place(0, ws.y.data());
                else if constexpr (!kIsDense) {
                    if (moe_ws != nullptr) {
                        Qwen3_5MoeMlpWorkspace& mw = *moe_ws;
                        if (nm.field == "router")
                            place(0, mw.router_logits.data());
                        else if (nm.field == "shared_expert.gate_up")
                            place(0, mw.shared_gate_up.data());
                        else if (nm.field == "shared_expert.down")
                            place(0, mw.shared_out.data());
                    }
                }
                break;
            }
            case PieForwardOpKind::SplitQkv:
                place(0, la.fa_qg_packed.data());
                place(1, ws.k.data());
                place(2, ws.v.data());
                break;
            case PieForwardOpKind::SplitGdn:
                // The two splits are `Launch` now (2b), so their pin
                // entries live in the Launch case below, keyed by
                // SYMBOL rather than by a width comparison. This one
                // stays for a semantic trace and does nothing for a
                // CUDA one, which no longer carries the kind -- the
                // same shape the norms' entries took in 1a, and the
                // same reason: an entry left keyed on a kind the class
                // trace stopped emitting stops applying silently.
                break;
            case PieForwardOpKind::GdnPrep:
                place(0, la.q_pre.data());
                place(1, la.k_pre.data());
                place(2, la.v_fp32.data());
                place(3, la.g_log.data());
                place(4, la.beta.data());
                break;
            case PieForwardOpKind::RmsnormGated:
                place(0, la.core_out_bf16.data());
                break;
            case PieForwardOpKind::SplitQGate:
                place(0, ws.q.data());
                place(1, la.fa_gate.data());
                break;
            case PieForwardOpKind::RmsnormPerHead: {
                // ONE output, and which buffer it is depends on the
                // weight -- lumping this with rope pinned a k_norm's
                // result to `ws.q`, which is the whole of what a pin
                // table is for getting right.
                const ParsedWeightName nm =
                    parse_weight_name(plan.weight_name(op));
                place(0, nm.field == "k_norm" ? ws.k.data() : ws.q.data());
                break;
            }
            case PieForwardOpKind::Rope:
                // Two outputs, rewritten where they lie; the trace
                // states the aliases, so these name the buffers their
                // operands already sit in.
                place(0, ws.q.data());
                place(1, ws.k.data());
                break;
            case PieForwardOpKind::SigmoidGateMul:
                place(0, ws.attn_out.data());
                break;
            case PieForwardOpKind::Swiglu:
                place(0, ws.gate.data());
                break;
            case PieForwardOpKind::LmHead:
                place(0, ws.logits.data());
                break;
            case PieForwardOpKind::Guard:
                break;  // see the fallback pass below
            case PieForwardOpKind::Launch: {
                switch (declared::resolve_kernel(plan.weight_name(op))) {
                case declared::Kernel::SplitRows:
                    // `[conv_dim | V_dim]` -- the qkvz row split.
                    place(0, la.mixed_qkv.data());
                    place(1, la.z.data());
                    break;
                case declared::Kernel::SplitGdnBa:
                    // The interleaved b/a split.
                    place(0, la.b.data());
                    place(1, la.a.data());
                    break;
                case declared::Kernel::RopeFull:
                case declared::Kernel::RopePartial:
                    // The rotation rewrites q and k where they lie; the
                    // pins follow the statement, exactly as they did
                    // when this was the semantic kind above.
                    place(0, ws.q.data());
                    place(1, ws.k.data());
                    break;
                case declared::Kernel::RmsnormRow:
                case declared::Kernel::RmsnormRowGemma:
                    // All three sites -- attn_norm, mlp_norm,
                    // final_norm -- land in `norm_x`. qwen3_5's
                    // post-attention norm reads it where llama_like
                    // reads `norm_y`.
                    place(0, ws.norm_x.data());
                    break;
                case declared::Kernel::AttnFlashinferDecode:
                case declared::Kernel::AttnFlashinferPrefill:
                    place(0, ws.attn_out.data());
                    break;
                case declared::Kernel::ConvUpdateBatched:
                case declared::Kernel::ConvPrefillBatched:
                    place(0, la.mixed_qkv_post.data());
                    break;
                case declared::Kernel::StepBatched:
                case declared::Kernel::StepBatchedBf16:
                case declared::Kernel::StepBatchedGqa:
                case declared::Kernel::StepBatchedGqaBf16:
                case declared::Kernel::PrefillWarpTiledGqa:
                case declared::Kernel::PrefillWarpTiledGqaBf16:
                case declared::Kernel::PrefillCached:
                case declared::Kernel::PrefillCachedBf16:
                case declared::Kernel::PrefillFla:
                case declared::Kernel::PrefillFlaBf16:
                    // Ten spellings of the recurrence, one result: the
                    // core the gated norm reads. Which one a fire takes
                    // is the guard's business and not this table's.
                    place(0, la.core_out.data());
                    break;
                case declared::Kernel::Swiglu:
                case declared::Kernel::ChunkedSwiglu:
                    // The dense MLP's activation, whichever spelling the
                    // binding took -- both land in `ws.gate`, which is
                    // what `down` reads. The routed and shared-expert
                    // callers of the same kernel write the MoE
                    // workspace and get entries when that island moves.
                    if constexpr (kIsDense) {
                        place(0, ws.gate.data());
                    }
                    break;
                default:
                    // The GDN recurrence, the conv, the MoE leg and the
                    // MTP stash still name their own buffers in their own
                    // arms; they get entries when those islands move.
                    break;
                }
                break;
            }
            default:
                break;
            }
        }
    }

    // THE GUARDS, and why they come second.
    //
    // A guard never executes: `lower()` resolves the chain and only the
    // winning region's launches appear, so a guard's result is written
    // by that region. Where the region's own launch declares the value,
    // ITS entry above is the authority and this must not touch it --
    // which is the bug this pass replaced. Both of this family's guards
    // produce a `[rows, 2048]` result, so a blanket entry here was
    // indistinguishable by width and clobbered the recurrence core's
    // pin with the attention output's buffer.
    //
    // What is left is the case that needs an entry at all, and there are
    // TWO of them: several launches in these chains declare no outputs
    // -- the attention dispatches, and the recurrence's prefill
    // spellings -- so their guard's result has no producer to inherit
    // from. Both results are `[rows, 2048]`, so the width cannot tell
    // them apart and a blanket entry gave the recurrence core the
    // attention's buffer.
    //
    // ONE of the two is computable from the guard and one is not, and
    // the difference is worth stating.
    //
    // The RECURRENCE's prefill spellings sit in a value-producing guard
    // whose regions are exactly those spellings, so the span answers
    // for them -- see `binds` below. The ATTENTION dispatch sits in no
    // guard at all: its result is simply a value nothing declares. That
    // is the gap, and it is narrower than it first looked.
    //
    // The CONSUMER can. A recurrence core is what the gated norm reads;
    // an attention result is what the output gate reads. That is a
    // statement of the convention rather than a heuristic -- those two
    // arms are the only readers either chain has -- and it is exactly
    // the kind of thing this table exists to record until the guard
    // states where its regions land.
    {
        const std::size_t op_count = plan.op_count();
        for (std::size_t i = 0; i < op_count; ++i) {
            const PieForwardOp& op = plan.op(i);
            if (op.kind != PieForwardOpKind::Guard) continue;
            for (const std::uint32_t v : plan.outputs(op)) {
                if (values.is_pinned(v)) continue;
                void* home = nullptr;
                for (std::size_t j = i + 1; j < op_count && home == nullptr;
                     ++j) {
                    const PieForwardOp& r = plan.op(j);
                    bool reads = false;
                    for (const std::uint32_t in : plan.inputs(r)) {
                        if (in == v) reads = true;
                    }
                    if (!reads) continue;
                    if (r.kind == PieForwardOpKind::RmsnormGated) {
                        home = la.core_out.data();
                    } else if (r.kind == PieForwardOpKind::SigmoidGateMul) {
                        home = ws.attn_out.data();
                    }
                }
                if (home != nullptr) values.pin(v, home);
            }
        }
    }

    // WHICH VALUE AN OP BINDS: the enclosing value-producing guard's
    // result.
    //
    // `Guard` is the one construct whose result has more than one
    // writer, and the ABI says so -- "the guard's outputs are the ONE
    // producer whichever region runs; region launches bind the same
    // output buffer and record no outputs of their own". That is an SSA
    // phi, and deliberate: an arm recording its own output would give
    // the value two definitions.
    //
    // Regions are FLAT and CONSECUTIVE -- `param0` arms, `[kind,
    // payload, len]` each plus a trailing else length in the aux run --
    // so the span is computable, and every op inside it binds the
    // result. Guards NEST (llama_like's outer body guard contains
    // three), so an inner guard's own span wins where they overlap;
    // this walks outermost-first and lets later, narrower writes
    // overwrite.
    std::vector<std::uint32_t> binds(plan.op_count(),
                                     pie_forward::PIE_FORWARD_NO_VALUE);
    for (std::size_t i = 0; i < plan.op_count(); ++i) {
        const PieForwardOp& g = plan.op(i);
        if (g.kind != PieForwardOpKind::Guard) continue;
        const auto gouts = plan.outputs(g);
        if (gouts.size == 0) continue;  // a branch that produces nothing
        const auto run = plan.aux_names(g);
        const std::uint32_t arms = g.param0;
        if (run.size < static_cast<std::size_t>(arms) * 3 + 1) continue;
        std::size_t span = 0;
        for (std::uint32_t a = 0; a < arms; ++a) span += run[a * 3 + 2];
        span += run[arms * 3];
        for (std::size_t j = i + 1; j <= i + span && j < binds.size(); ++j) {
            binds[j] = gouts[0];
        }
    }

    // An arm indexes operands positionally, and a span SHORTER than the
    // arm assumes is not a crash — it reads the next statement's
    // operands and hands the arm a plausible pointer to the wrong
    // buffer.
    const auto need = [&](const auto& span, std::size_t n, const char* what) {
        if (span.size < n) {
            throw std::runtime_error(
                std::string("declared qwen35: ") + what + " states " +
                std::to_string(span.size) + " operands, needs " +
                std::to_string(n));
        }
    };

    // A value's trailing dims ARE its row width — which is how `conv_dim`,
    // `V_dim`, `V_h`, `qgkv_dim`, `2 * Hq`, `Hk`, `I` and the rest stop
    // being per-branch constants an arm has to know to pick a buffer.
    const auto row_width = [&](std::uint32_t id) {
        const auto& val = plan.value(id);
        std::uint32_t out = 1;
        for (std::uint32_t d = 1; d < val.rank; ++d) {
            if (val.dims[d].kind != pie_forward::PieForwardDimKind::Const) {
                return 0;
            }
            out *= val.dims[d].value;
        }
        return static_cast<int>(out);
    };

    // `PIE_Q35_PIN_AUDIT=1`: with `PIE_DECLARED_HOST_ARENA=0` nothing
    // may move, so EVERY operand a converted arm touches must answer to
    // the pin table. One that does not is not a fault -- it is a
    // plausible pointer to bytes no unconverted arm writes -- so it
    // needs a report rather than a crash. This is the generalisation of
    // the probe that found the dense swiglu's missing entry.
    const bool pin_audit = std::getenv("PIE_Q35_PIN_AUDIT") != nullptr;
    const auto audit = [&](const PieForwardOp& op) {
        static std::set<std::string> seen;
        int idx = -1;
        const auto look = [&](std::uint32_t v, const char* side) {
            ++idx;
            if (values.is_pinned(v)) return;
            std::string key = std::to_string(
                                  static_cast<std::uint32_t>(op.kind)) +
                              side + std::string(plan.weight_name(op));
            if (!seen.insert(key).second) return;
            // Name the PRODUCER too: a missing entry is fixed where the
            // value is written, not where it is read.
            std::string producer = "(none)";
            for (std::size_t j = 0; j < plan.op_count(); ++j) {
                const PieForwardOp& q = plan.op(j);
                bool writes = false;
                for (const std::uint32_t o : plan.outputs(q)) {
                    if (o == v) writes = true;
                }
                if (!writes) continue;
                producer = std::to_string(
                               static_cast<std::uint32_t>(q.kind)) +
                           ":" + std::string(plan.weight_name(q));
                break;
            }
            std::fprintf(stderr,
                         "[q35-pin-audit] kind=%u %s v%u UNPINNED '%.*s' "
                         "written by %s  [operand %d, width %d]\n",
                         static_cast<std::uint32_t>(op.kind), side, v,
                         static_cast<int>(plan.weight_name(op).size()),
                         plan.weight_name(op).data(), producer.c_str(),
                         idx, row_width(v));
        };
        idx = -1;
        for (const std::uint32_t v : plan.inputs(op)) look(v, "in");
        idx = -1;
        for (const std::uint32_t v : plan.outputs(op)) look(v, "out");
    };

    const bool ext_dump = std::getenv("PIE_Q35_EXTENT_DUMP") != nullptr;
    const auto extents = [&](const PieForwardOp& op) {
        static std::set<std::string> seen;
        std::string key = std::to_string(
                              static_cast<std::uint32_t>(op.kind)) +
                          ":" + std::string(plan.weight_name(op));
        if (!seen.insert(key).second) return;
        std::string line = "[q35-ext] kind=" +
                           std::to_string(
                               static_cast<std::uint32_t>(op.kind)) +
                           " '" + std::string(plan.weight_name(op)) +
                           "' p0=" + std::to_string(op.param0) +
                           " p1=" + std::to_string(op.param1) + "  in:";
        for (const std::uint32_t v : plan.inputs(op)) {
            line += " v" + std::to_string(v) + "/w" +
                    std::to_string(row_width(v));
        }
        line += "  out:";
        for (const std::uint32_t v : plan.outputs(op)) {
            line += " v" + std::to_string(v) + "/w" +
                    std::to_string(row_width(v));
        }
        std::fprintf(stderr, "%s\n", line.c_str());
    };

    const auto execute_op = [&](const PieForwardOp& op, std::size_t at_op) {
        // Where a statement's result lands: its own declared output if
        // it has one (the decode recurrence states it), else the value
        // its enclosing guard owns.
        // The attention's destination. Its launches DO declare an
        // output in both classes -- checked against the goldens, which
        // corrects an earlier reading here that they did not -- so it is
        // the statement's value, with the guard's as the fallback and
        // `ws.attn_out` behind that.
        // What a shared arm is handed when the statement declares no
        // result: the enclosing value-producing guard's value, never the
        // statement's own result. The arm reads that itself, and asking
        // for `plan.outputs(op)[0]` here would ask it of every op the
        // walk sees rather than of the dispatches — a slot lookup on a
        // value that may be the backend's to bind, for no reason.
        const auto region_dst = [&]() -> void* {
            const std::uint32_t b =
                at_op < binds.size() ? binds[at_op]
                                     : pie_forward::PIE_FORWARD_NO_VALUE;
            if (b != pie_forward::PIE_FORWARD_NO_VALUE) return values.slot(b);
            return ws.attn_out.data();
        };
        const auto bound_or_out = [&]() -> void* {
            const auto o = plan.outputs(op);
            if (o.size > 0) return values.slot(o[0]);
            const std::uint32_t b =
                at_op < binds.size() ? binds[at_op]
                                     : pie_forward::PIE_FORWARD_NO_VALUE;
            if (b != pie_forward::PIE_FORWARD_NO_VALUE) return values.slot(b);
            throw_drift("a launch that declares no output sits in no "
                        "value-producing guard");
        };
        // The recurrence's five operands, all five the statement's.
        //
        // q and k were the exception until the repeat declared its
        // result: the GQA head repeat sits between the prep and the
        // cached recurrence, and while it was output-less the value it
        // produced had no id, so the recurrence's stated q/k still named
        // the PRE-repeat pair and the arm had to substitute a workspace
        // buffer for both. The cached arm now states the repeat's
        // results as the recurrence's operands, which is what the arm
        // always meant, so this reads the span and nothing else.
        const auto rec_in = [&](std::size_t i) -> const float* {
            const auto ins = plan.inputs(op);
            need(ins, 5, "recurrence inputs");
            return static_cast<const float*>(values.slot(ins[i]));
        };
        const auto rec_q = [&]() -> const float* { return rec_in(0); };
        const auto rec_k = [&]() -> const float* { return rec_in(1); };
        if (pin_audit) audit(op);
        if (ext_dump) extents(op);
        switch (op.kind) {
        case PieForwardOpKind::Embed: {
            const std::string_view name = plan.weight_name(op);
            if (name != "embed") throw_unknown_weight(name);
            // ISLAND (value arena). `token_ids` stays a driver input --
            // it is the fire's, not a traced value.
            const auto outs = plan.outputs(op);
            need(outs, 1, "embed outputs");
            declared::arm_embed({plan, values, N, 0, stream}, op, token_ids, wb.require(name).data(), cfg.vocab_size);
            break;
        }
        case PieForwardOpKind::Rmsnorm:
            // RUNG 5: the semantic cascade is deleted -- a class
            // trace states which FOLD it runs (`cuda::rmsnorm`),
            // so this kind reaching the walk means the trace and
            // this executor drifted. Choosing here from a param
            // is what the statement now says instead.
            throw_drift("semantic Rmsnorm in a class trace "
                    "(the declaration states the fold)");

        case PieForwardOpKind::Matmul: {
            // ISLAND (value arena). Fifteen branches keyed on the weight
            // NAME chose a buffer pair and three extents each; every one
            // of those is the statement's, and only the WEIGHT side is
            // not.
            //
            // So the name dispatch stays, shrunk to what it is actually
            // for: this family stores several projections QUANTIZED, and
            // which quant descriptor a weight carries is a per-field
            // fact (`fa_q_proj_quant`, `down_proj_quant`, ...) that no
            // value descriptor states. `M`, `N`, `K` and both buffers
            // come off the trace -- `conv_dim`, `V_dim`, `V_h`,
            // `qgkv_dim`, `2 * Hq`, `Hk`, `I` and `H` were the executor
            // knowing by convention what the values already say.
            const std::string_view name = plan.weight_name(op);
            const ParsedWeightName nm = parse_weight_name(name);
            const auto& layer = layer_of(w, nm, name);
            const float beta = op.param0 != 0 ? 1.f : 0.f;
            const auto ins = plan.inputs(op);
            const auto outs = plan.outputs(op);
            need(ins, 1, "matmul inputs");
            need(outs, 1, "matmul outputs");
            const int M = N;
            const int cols = row_width(outs[0]);
            const int depth = row_width(ins[0]);
            // `PIE_Q35_MATMUL_DUMP=1`: one line per distinct projection,
            // its extents off the trace and whether each operand came
            // from the PIN table or the arena.
            //
            // The `in_pin` column is the one that earned this: with pins
            // forced everywhere the island still diverged, and the cause
            // was `down` reading an operand no pin covered -- the dense
            // swiglu is a `Launch` in this family, not the semantic
            // kind, so the pin switch had missed it. An unpinned operand
            // mid-migration is not a fault; it is a plausible pointer to
            // bytes no unconverted arm writes, which is why it needs a
            // column rather than a crash.
            if (std::getenv("PIE_Q35_MATMUL_DUMP") != nullptr) {
                static std::set<std::string> seen;
                std::string key(nm.field);
                if (seen.insert(key).second) {
                    std::fprintf(stderr,
                                 "[q35-matmul] %-24s M=%d cols=%d depth=%d "
                                 "beta=%.0f in_pin=%d out_pin=%d\n",
                                 key.c_str(), M, cols, depth, beta,
                                 values.is_pinned(ins[0]) ? 1 : 0,
                                 values.is_pinned(outs[0]) ? 1 : 0);
                }
            }

            // The PACKED bank, and only that (2d). An unfused binding
            // states its two halves as two matmuls, each with its own
            // weight name and its own traced result -- so this branch no
            // longer fires two GEMMs for one statement into buffers the
            // traced value did not describe, and the
            // `gate_up_used_fused` correspondence with the activation
            // arm goes with it.
            if (nm.field == "gate_up") {
                if constexpr (kIsDense) {
                    kernels::gemm::act_x_w(cublas.handle(),
                        values.slot(ins[0]),
                        WeightView(*require(layer.gate_up_proj_fused, name)),
                        values.slot(outs[0]), M, cols, depth);
                } else { throw_unknown_weight(name); }
                break;
            }
            if (nm.field == "gate_proj" || nm.field == "up_proj") {
                if constexpr (kIsDense) {
                    kernels::gemm::act_x_w(cublas.handle(),
                        values.slot(ins[0]),
                        dense(wb.require(name),
                              nm.field == "gate_proj" ? layer.gate_proj_quant
                                                      : layer.up_proj_quant,
                              name),
                        values.slot(outs[0]), M, cols, depth);
                } else { throw_unknown_weight(name); }
                break;
            }

            // Everything else is one gemm; the name says only which
            // quant descriptor rides with the weight.
            const WeightView wv = [&]() -> WeightView {
                if (nm.field == "q_proj")
                    return dense(wb.require(name), layer.fa_q_proj_quant,
                                 name);
                if (nm.field == "k_proj")
                    return dense(wb.require(name), layer.fa_k_proj_quant,
                                 name);
                if (nm.field == "v_proj")
                    return dense(wb.require(name), layer.fa_v_proj_quant,
                                 name);
                if (nm.field == "o_proj") {
                    // A linear-attention layer's o_proj is never
                    // quantized in this family; a full-attention one may
                    // be. `layer.kind` is what tells them apart, exactly
                    // as it did when the branch also picked the input
                    // buffer.
                    return layer.kind == LayerW::Kind::LinearAttn
                               ? WeightView(wb.require(name))
                               : dense(wb.require(name),
                                       layer.fa_o_proj_quant, name);
                }
                // The two layer structs carry DIFFERENT quant fields --
                // `down_proj_quant` is the dense MLP's, and only a MoE
                // layer has a shared expert to quantize -- so each is
                // reached under the fence that makes it exist.
                if constexpr (kIsDense) {
                    if (nm.field == "down")
                        return dense(wb.require(name),
                                     layer.down_proj_quant, name);
                } else {
                    if (nm.field == "shared_expert.down")
                        return dense(wb.require(name),
                                     layer.shared_down_proj_quant, name);
                }
                if (nm.field == "in_proj_qkv" || nm.field == "in_proj_z" ||
                    nm.field == "in_proj_a" || nm.field == "in_proj_b" ||
                    nm.field == "qgkv" || nm.field == "router" ||
                    nm.field == "shared_expert.gate_up") {
                    return WeightView(wb.require(name));
                }
                throw_unknown_weight(name);
            }();
            if constexpr (!kIsDense) {
                if (nm.field == "router" ||
                    nm.field == "shared_expert.gate_up" ||
                    nm.field == "shared_expert.down") {
                    if (moe_ws == nullptr) {
                        throw_drift("the MoE leg needs its workspace");
                    }
                }
            }
            kernels::gemm::act_x_w(cublas.handle(), values.slot(ins[0]), wv,
                                   values.slot(outs[0]), M, cols, depth, beta);
            break;
        }
        case PieForwardOpKind::SplitQkv: {
            // Fused full-attn bank split: the "q" leg is the 2×-wide
            // [query | gate] pack (`use_fused_qgkv` in the hand-written
            // body: kernels::attn::split_qkv_bf16(packed, qg, k, v, N, 2*Hq, Hk)).
            // ISLAND (value arena). `2 * Hq` and `Hk` are the two
            // result widths, which the results state.
            // SHARED ARM (D1) -- see gemma-4's call site.
            declared::arm_split_qkv({plan, values, N, 0, stream}, op);
            break;
        }
        case PieForwardOpKind::SplitGdn:
            // RUNG 5: the width comparison is deleted -- a class trace
            // names which split it runs (`cuda::split_rows` /
            // `cuda::split_qwen_gdn_ba`).
            throw_drift("semantic SplitGdn reached the class-trace walk "
                        "(the declaration states the split)");
        case PieForwardOpKind::CausalConv1d: {
            // RUNG 5: the semantic cascade is deleted — a class trace
            // states this choice site's kernels.
            throw_drift("semantic CausalConv1d reached the class-trace walk "
                        "(the declaration states the conv kernel)");
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
            // ISLAND (value arena). Three operands in, five results
            // out, all the statement's. The HEAD GEOMETRY stays read
            // from config: `K_h`, `V_h`, `K_d`, `V_d` are how the
            // recurrence carves a row, and a row width divided by a head
            // count is that carving only once you already know one of
            // them -- which this op does not state.
            const auto ins = plan.inputs(op);
            const auto outs = plan.outputs(op);
            need(ins, 3, "gdn_prep inputs");
            need(outs, 5, "gdn_prep outputs");
            kernels::ssm::qwen_gdn_post_conv_prep_bf16(
                values.slot(ins[0]), values.slot(ins[1]),
                values.slot(ins[2]),
                layer.la_A_log_fp32,
                require(layer.la_dt_bias, dt_name)->data(),
                static_cast<float*>(values.slot(outs[0])),
                static_cast<float*>(values.slot(outs[1])),
                static_cast<float*>(values.slot(outs[2])),
                static_cast<float*>(values.slot(outs[3])),
                static_cast<float*>(values.slot(outs[4])),
                N, K_h, V_h, K_d, V_d, conv_dim, stream);
            // GQA materialisation is a LOWERING of the recurrence, not a
            // trace op: the decode GQA step, warp-tiled prefill and
            // batched-FLA-GQA kernels all index the compact K_h-head
            // layout directly, so repeat_interleave launches only when
            // none of them is eligible — the hand-written predicate,
            // all four terms.
            // RUNG 5: the GQA repeat derivation is deleted — a class
            // trace STATES the repeats inside the recurrence guard's
            // cached arm, and nowhere else.
            break;
        }
        case PieForwardOpKind::GatedDelta: {
            // RUNG 5: the semantic cascade is deleted — a class trace
            // states this choice site's kernels.
            throw_drift("semantic GatedDelta reached the class-trace walk "
                        "(the declaration states the recurrence)");
        }
        case PieForwardOpKind::RmsnormGated: {
            // core_out (fp32) → fused z-gated RMSNorm → bf16, per (n, h)
            // row of V_d — the hand-written fused kernel, one launch.
            const std::string_view name = plan.weight_name(op);
            const ParsedWeightName nm = parse_weight_name(name);
            if (nm.field != "gate_norm") throw_unknown_weight(name);
            const auto& layer = layer_of(w, nm, name);
            if (layer.la_norm_w_fp32 == nullptr) throw_unknown_weight(name);
            // ISLAND (value arena).
            const auto ins = plan.inputs(op);
            const auto outs = plan.outputs(op);
            need(ins, 2, "gated norm inputs");
            need(outs, 1, "gated norm outputs");
            kernels::norm::rmsnorm_gated_fp32_in_bf16(
                values.slot(ins[0]), values.slot(ins[1]),
                layer.la_norm_w_fp32, values.slot(outs[0]),
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
            // ISLAND (value arena). The geometry is checked against the
            // params above and stays config's, for `GdnPrep`'s reason.
            const auto ins = plan.inputs(op);
            const auto outs = plan.outputs(op);
            need(ins, 1, "split_q_gate inputs");
            need(outs, 2, "split_q_gate outputs");
            kernels::layout::split_q_gate_bf16(
                values.slot(ins[0]), values.slot(outs[0]),
                values.slot(outs[1]), N, num_q_heads, d, stream);
            break;
        }
        case PieForwardOpKind::RmsnormPerHead: {
            // Gemma fold, in place, one row per head — the hand-written
            // q/k norms (`kernels::norm::rmsnorm_gemma_bf16` over N·heads rows).
            if (op.param1 !=
                static_cast<std::uint32_t>(PieForwardNormVariant::Gemma)) {
                throw_drift("only the Gemma per-head norm is emitted");
            }
            const std::string_view name = plan.weight_name(op);
            const ParsedWeightName nm = parse_weight_name(name);
            const auto& layer = layer_of(w, nm, name);
            // ISLAND (value arena). Two sites that differed in which
            // buffer they normed and how many HEAD-WIDE rows that is --
            // both the statement's. `op.param0` is the head width, so
            // the row count is the operand's width divided by it, and
            // the arm stops needing to know which site it is in.
            //
            // The convention passed one pointer twice. That is the
            // CONVENTION choosing to overwrite, not the kernel needing
            // to -- it computes correctly into a fresh buffer, and the
            // declaration says out and in are different values. So this
            // does NOT claim an alias the way rope's arm does.
            const auto ins = plan.inputs(op);
            const auto outs = plan.outputs(op);
            need(ins, 1, "per-head norm inputs");
            need(outs, 1, "per-head norm outputs");
            const int head = static_cast<int>(op.param0);
            if (head <= 0) throw_drift("per-head norm states no head width");
            kernels::norm::rmsnorm_gemma_bf16(
                values.slot(ins[0]), wb.require(name).data(),
                values.slot(outs[0]), N * (row_width(ins[0]) / head), head,
                eps, stream);
            break;
        }
        case PieForwardOpKind::Rope:
            // RUNG 5: the width branch is deleted -- a class trace
            // states which rotation it runs (`cuda::rope` /
            // `cuda::rope_partial`).
            throw_drift("semantic Rope reached the class-trace walk "
                        "(the declaration states the rotation)");
        case PieForwardOpKind::KvAppend: {
            // RUNG 5: the semantic cascade is deleted — a class trace
            // states this choice site's kernels.
            throw_drift("semantic KvAppend reached the class-trace walk "
                        "(the declaration states the KV write)");
        }
        case PieForwardOpKind::Attention: {
            // RUNG 5: the semantic cascade is deleted — a class trace
            // states this choice site's kernels.
            throw_drift("semantic Attention reached the class-trace walk "
                        "(the declaration states the attention kernel)");
        }
        case PieForwardOpKind::SigmoidGateMul: {
            // attn_out *= sigmoid(gate) — the full-attention output gate.
            // ISLAND (value arena). In place over operand 0, which the
            // trace now states (`kernels::semantic_in_place`) -- the
            // kernel's own name says so and it has no destination to
            // give it another.
            const auto ins = plan.inputs(op);
            const auto outs = plan.outputs(op);
            need(ins, 2, "sigmoid gate inputs");
            need(outs, 1, "sigmoid gate outputs");
            kernels::mlp::sigmoid_gate_inplace_bf16(
                values.slot(outs[0]), values.slot(ins[1]),
                N * row_width(outs[0]), stream);
            break;
        }
        case PieForwardOpKind::Swiglu:
            // RUNG 5: a class trace states which activation runs, and
            // (2d) the operands it reads.
            throw_drift("semantic Swiglu reached the class-trace walk "
                        "(the declaration states the activation)");
case PieForwardOpKind::Launch: {
            // The dumb arm (rung 4c-iii): resolve the STATED launcher
            // symbol and bind. Each handler is the corresponding branch
            // of the semantic cascade, minus the choosing; the state
            // layer rides param1 (RecurrentState store for the GDN
            // kernels, the MODEL layer for KV-side ones — the compact
            // kv slot derives from the binding, mechanical knowledge).
            const int SL = static_cast<int>(op.param1);
            const auto conv_weight = [&]() -> const LayerW& {
                const auto aux = plan.aux_names(op);
                if (aux.size != 1) {
                    throw_drift("conv launch names " +
                                std::to_string(aux.size) +
                                " weights, wants 1");
                }
                const std::string_view nm_s = plan.name(aux[0]);
                const ParsedWeightName nm = parse_weight_name(nm_s);
                if (nm.field != "conv") throw_drift("conv launch weight");
                return layer_of(w, nm, nm_s);
            };
            const auto kv_view_of = [&](int model_layer) {
                if (model_layer < 0 ||
                    model_layer >= static_cast<int>(w.layers.size()) ||
                    w.layers[model_layer].kv_layer < 0) {
                    throw_drift("launch layer " +
                                std::to_string(model_layer) +
                                " has no KV cache slot");
                }
                return cache.layer_view(w.layers[model_layer].kv_layer);
            };
            void* const rs_slot0 =
                op.param0 == 2  // RecurrentState store mark
                    ? state_cache.recurrent_state_raw(SL, /*slot=*/0)
                    : nullptr;
            // THE SHARED SWITCH FIRST (D1). Every symbol whose arm is
            // family-blind lives in `declared/execute.hpp`; what remains
            // below is this family's RESIDUE. A `false` is an answer --
            // "stated, and this family executes it its own way".
            const declared::ExecCtx ectx{
                {plan, values, N, 0, stream},
                wb, cache, attn_ws, cublas, /*tp_comm=*/nullptr,
                &state_cache,
                positions, qo_indptr, kv_page_indices, kv_page_indptr,
                kv_last_page_lens,
                // NULL, deliberately: this family's KV write arms passed
                // no row-valid mask, and the merge must not start
                // skipping rows they wrote. Whether they SHOULD pass it
                // is a question for the family, not for a refactor.
                nullptr,
                qo_indptr_h, kv_page_indptr_h,
                w_page_d, w_off_d, R,
                nullptr, false,
                eps, /*sm_scale=*/-1.f, /*lse_fallback=*/nullptr,
                cfg.rope_theta,
                num_q_heads, num_kv_heads, d, d,
                // The CACHE SLOT, not the model layer: only this
                // family's full-attention layers have one.
                (SL >= 0 && SL < static_cast<int>(w.layers.size()))
                    ? w.layers[static_cast<std::size_t>(SL)].kv_layer
                    : -1,
                // The plans the prepare built for this fire, and the
                // destination for a dispatch whose enclosing guard owns
                // the value. Both were this family's own attention arms
                // a moment ago; the arms are one now, and this is what
                // it took.
                decode_plan, prefill_plan, region_dst(),
                // The recurrent state's window for this launch. The slab
                // is the layer the statement marks; the rest is the
                // fire's.
                rs_slot0, slot_ids_d,
                static_cast<long long>(slot_stride), write_state,
                commit_lens,
            };
            if (declared::execute_shared(ectx, op)) break;
            switch (declared::resolve_kernel(plan.weight_name(op))) {
            case declared::Kernel::SplitRows:
            case declared::Kernel::SplitGdnBa: {
                // The two GDN splits, told apart by their SYMBOLS. The
                // arm they replace compared `op.param0`/`param1`
                // against `conv_dim`, `V_dim` and `V_h` -- which works
                // only while those three stay distinct, and is a kernel
                // chosen from a coincidence of extents either way.
                const auto si = plan.inputs(op);
                const auto so = plan.outputs(op);
                need(si, 1, "split inputs");
                need(so, 2, "split outputs");
                if (declared::resolve_kernel(plan.weight_name(op)) ==
                    declared::Kernel::SplitRows) {
                    kernels::layout::split_bf16_rows(
                        values.slot(si[0]), values.slot(so[0]),
                        values.slot(so[1]), N, row_width(so[0]),
                        row_width(so[1]), stream);
                } else {
                    kernels::layout::split_qwen_gdn_ba_bf16(
                        values.slot(si[0]), values.slot(so[0]),
                        values.slot(so[1]), N, row_width(so[0]), stream);
                }
                break;
            }
            case declared::Kernel::ConvUpdateBatched: {
                const auto& layer = conv_weight();
                // ISLAND (value arena). The conv state, the slot ids
                // and the stride stay the CACHE's -- a per-request
                // recurrent slot is not a traced value.
                const auto ins = plan.inputs(op);
                const auto outs = plan.outputs(op);
                need(ins, 1, "conv inputs");
                need(outs, 1, "conv outputs");
                kernels::ssm::causal_conv1d_update_batched_bf16(
                    values.slot(ins[0]), layer.la_conv1d_w->data(),
                    layer.la_conv1d_b ? layer.la_conv1d_b->data() : nullptr,
                    state_cache.conv_state(SL, /*slot=*/0),
                    slot_ids_d,
                    static_cast<long long>(state_cache.conv_kernel()) *
                        state_cache.conv_dim(),
                    values.slot(outs[0]),
                    R, conv_dim, conv_K, stream);
                break;
            }
            case declared::Kernel::ConvPrefillBatched: {
                const auto& layer = conv_weight();
                // ISLAND (value arena).
                const auto ins = plan.inputs(op);
                const auto outs = plan.outputs(op);
                need(ins, 1, "conv inputs");
                need(outs, 1, "conv outputs");
                kernels::ssm::causal_conv1d_prefill_batched_bf16(
                    values.slot(ins[0]), layer.la_conv1d_w->data(),
                    layer.la_conv1d_b ? layer.la_conv1d_b->data() : nullptr,
                    values.slot(outs[0]),
                    state_cache.conv_state(SL, /*slot=*/0),
                    slot_ids_d, qo_indptr,
                    static_cast<long long>(state_cache.conv_kernel()) *
                        state_cache.conv_dim(),
                    R, conv_dim, conv_K, stream, write_state,
                    commit_lens);
                break;
            }
            // THE RECURRENCE, ten spellings, all GENERATED. What kept
            // them here was not arithmetic -- the five operands were
            // already the statement's -- but four things around them:
            // the recurrent-state slab, the request slot addressing,
            // the frozen-verify write suppression and the spec-decode
            // commit lengths. All four are the FIRE's, so they follow
            // `kv_layer` and the attention plan into the context and the
            // family hands them over.
            //
            // The head geometry comes off the two VALUES: q is
            // `[Tokens, heads, key_dim]` and v is `[Tokens, value_heads,
            // value_dim]`, which is every count these kernels take. And
            // the destination is `ResultOrRegion`: the decode step
            // declares its result, the six prefill spellings are arms of
            // a value-producing guard and declare none.
            // The head repeat GENERATES, which it could not while it
            // was output-less: all three counts are dims of the two
            // values -- `[Tokens, key_heads, key_dim]` in, `[Tokens,
            // value_heads, key_dim]` out -- so the row names them and
            // the arm derives.
            case declared::Kernel::VerifyStashLoad:
            case declared::Kernel::VerifyStashStore: {
                // The pseudo-symbols name an OPERATION the driver
                // implements as a cudaMemcpyAsync trio ([mixed_qkv|a|b]
                // against the layer's stash slab) — a launcher may be
                // three API calls; the symbol names the operation. The
                // stash is keyed by the COMPACT linear index, storage
                // knowledge derived from the binding (the semantic arm's
                // derivation, verbatim).
                if (!stash_enabled) {
                    throw_drift("stated stash op but the live stash is "
                                "disabled (cross-check should have "
                                "routed this fire to the semantic walk)");
                }
                int linear_idx = 0;
                for (int l = 0; l < SL; ++l) {
                    if (w.layers[l].kind ==
                        LayerW::Kind::LinearAttn) {
                        ++linear_idx;
                    }
                }
                auto* stash = static_cast<std::uint16_t*>(
                    state_cache.verify_hidden_stash_layer(linear_idx));
                const bool load =
                    declared::resolve_kernel(plan.weight_name(op)) ==
                    declared::Kernel::VerifyStashLoad;
                const auto cp = [&](void* dst, const void* src,
                                    std::size_t n) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        dst, src, n, cudaMemcpyDeviceToDevice, stream));
                };
                const std::size_t n_qkv =
                    static_cast<std::size_t>(N) * conv_dim *
                    sizeof(std::uint16_t);
                const std::size_t n_ab =
                    static_cast<std::size_t>(N) * V_h *
                    sizeof(std::uint16_t);
                // ISLAND (value arena), BOTH ENDS. The pair declares the
                // triple on both sides -- `verify_stash_load` states
                // three results, `verify_stash_store` three operands --
                // so the slab's peer is the statement's, and the arm no
                // longer restates which workspace field the in-proj
                // triple happens to live in. That was the DSL's own
                // deferral ("WHERE those buffers live is the driver's
                // binding") and the declaration outgrew it.
                const auto trip = load ? plan.outputs(op) : plan.inputs(op);
                if (trip.size != 3) {
                    throw_drift("a stash op states " +
                                std::to_string(trip.size) +
                                " halves of the in-proj triple, wants 3");
                }
                void* const v_qkv = values.slot(trip[0]);
                void* const v_a = values.slot(trip[1]);
                void* const v_b = values.slot(trip[2]);
                if (load) {
                    cp(v_qkv, stash, n_qkv);
                    cp(v_a, stash + stash_a_off, n_ab);
                    cp(v_b, stash + stash_b_off, n_ab);
                } else {
                    cp(stash, v_qkv, n_qkv);
                    cp(stash + stash_a_off, v_a, n_ab);
                    cp(stash + stash_b_off, v_b, n_ab);
                }
                break;
            }
            // The two flashinfer dispatches are SHARED now. What kept
            // them here was the plan, and the plan is the context's:
            // this family resolves it off `plan_state` and hands it
            // over, which is the only thing about the call that was
            // ever this family's.
            //
            // A null plan no longer throws here — the shared arm
            // returns false and the walk falls through to this switch,
            // where `default` refuses by name. That is the same answer
            // through one fewer copy of it.

            // The PAIR form (2d): two operands, both traced, so the
            // arm reads them off the plan. Only the dense MLP states
            // it -- the routed and shared legs always bind packed
            // banks and take the chunked kernel below.
            // The CHUNKED SWIGLU generates, and all three of its
            // callers with it -- the dense MLP's, the shared expert's
            // and the routed leg's.
            //
            // Three arms, because three answers to "how many rows".
            // Dense and shared run over tokens; the routed leg runs over
            // the padded block-major count, and this executor computed
            // that from a formula it kept beside four other copies. The
            // row says `OutRows` now, so the count comes off the value's
            // own leading extent and the three arms are one.
            // ── The aligned MoE leg ──────────────────────────────────
            //
            // MoE-only, so the whole group is fenced: a dense weights type
            // has no `moe_ws` to drive and no expert bank to bind. The
            // shapes are transcribed from `qwen3_5_moe_forward.cpp`'s
            // aligned block rather than re-derived -- this arm's job is to
            // fire the same launches, not to re-decide them.
            case declared::Kernel::TopkSoftmax:
            case declared::Kernel::MoeAlignDecode:
            case declared::Kernel::MoeGatherAligned:
            case declared::Kernel::MoeBuildPtrsAligned:
            case declared::Kernel::MoeGroupedGemm:
            case declared::Kernel::MoeReorderAligned:
            case declared::Kernel::MoeWeightedSum:
            case declared::Kernel::SigmoidDotScalarGateAdd: {
                if constexpr (kIsDense) {
                    throw_drift("a MoE launch in a dense fire");
                } else {
                    if (moe_ws == nullptr) {
                        throw_drift("the MoE leg needs its workspace");
                    }
                    Qwen3_5MoeMlpWorkspace& mw = *moe_ws;
                    const int E = cfg.num_experts;
                    const int Ktop = cfg.num_experts_per_tok;
                    const int Im = cfg.moe_intermediate_size;
                    const int routes = N * Ktop;
                    const int block = mw.aligned_block_size;
                    const int active_expert_cap = std::min(E, routes);
                    const int max_blocks =
                        (routes + active_expert_cap * (block - 1) + block - 1) /
                        block;
                    const int aligned_rows = max_blocks * block;
                    // The shared expert is NOT folded here, matching the
                    // hand path's `constexpr bool fold_shared = false`.
                    constexpr int shared_row_begin = -1;
                    switch (declared::resolve_kernel(plan.weight_name(op))) {
                    case declared::Kernel::TopkSoftmax:
                        // ISLAND (value arena). `topk(logits)` states
                        // one operand and two results.
                        kernels::moe::topk_softmax_bf16(
                            values.slot(plan.inputs(op)[0]),
                            static_cast<std::int32_t*>(
                                values.slot(plan.outputs(op)[0])),
                            static_cast<float*>(
                                values.slot(plan.outputs(op)[1])),
                            N, E, Ktop, stream);
                        break;
                    // The PERMUTATION generates. Its route count is
                    // the operand's element count -- `topk_idx` is
                    // `[Tokens, top_k]` -- and its three load-time
                    // numbers ride the param channel, where two came out
                    // of a config struct and one out of the MoE
                    // workspace. It binds the INVERSE MAP too, which the
                    // arm passed null for: the statement declares three
                    // results, and "declared but not written" is a claim
                    // the declaration does not make.
                    case declared::Kernel::MoeGatherAligned:
                        // ISLAND (value arena).
                        // `gather_moe_aligned_inputs(x, sorted_route_ids)`
                        // -- both operands and the result are stated.
                        kernels::moe::gather_moe_aligned_inputs_bf16(
                            values.slot(plan.inputs(op)[0]),
                            static_cast<const std::int32_t*>(
                                values.slot(plan.inputs(op)[1])),
                            values.slot(plan.outputs(op)[0]),
                            routes, aligned_rows, Ktop, H,
                            shared_row_begin, N, stream);
                        break;
                    case declared::Kernel::MoeBuildPtrsAligned: {
                        const auto aux = plan.aux_names(op);
                        if (aux.size != 2) {
                            throw_drift("the ptr build names " +
                                        std::to_string(aux.size) +
                                        " banks, wants 2");
                        }
                        // ISLAND (value arena). Two operands, three
                        // results: the build DECLARES the aligned
                        // staging, because it bakes those buffers' base
                        // addresses into the pointer arrays and so has to
                        // know where they are before anything writes
                        // them. Everything downstream takes its
                        // destination from here.
                        //
                        // The six POINTER ARRAYS are what is left. An
                        // array of device addresses has no dtype in the
                        // trace's vocabulary, so they stay the MoE
                        // workspace's -- reachable only from the two
                        // GEMMs this same call serves, which is what
                        // bounds the gap.
                        const auto bins = plan.inputs(op);
                        const auto bouts = plan.outputs(op);
                        need(bins, 2, "ptr build inputs");
                        need(bouts, 3, "ptr build outputs");
                        kernels::moe::build_moe_ptrs_aligned_bf16(
                            static_cast<const std::int32_t*>(
                                values.slot(bins[0])),
                            wb.require(plan.name(aux[0])).data(),
                            wb.require(plan.name(aux[1])).data(),
                            values.slot(bins[1]),
                            values.slot(bouts[0]),
                            values.slot(bouts[1]),
                            values.slot(bouts[2]),
                            reinterpret_cast<const void**>(mw.a_gu_ptrs.data()),
                            reinterpret_cast<const void**>(mw.b_gu_ptrs.data()),
                            reinterpret_cast<void**>(mw.c_gu_ptrs.data()),
                            reinterpret_cast<const void**>(mw.a_dn_ptrs.data()),
                            reinterpret_cast<const void**>(mw.b_dn_ptrs.data()),
                            reinterpret_cast<void**>(mw.c_dn_ptrs.data()),
                            max_blocks, block, H, Im,
                            /*shared_block_begin=*/max_blocks,
                            /*shared_gate_up=*/nullptr,
                            /*shared_down=*/nullptr, stream);
                        break;
                    }
                    case declared::Kernel::MoeGroupedGemm: {
                        // Which projection this is, read off the BANK the
                        // statement names -- not off a counter, which is how
                        // the two would drift once anything reorders them.
                        const auto aux = plan.aux_names(op);
                        if (aux.size != 1) {
                            throw_drift("the grouped GEMM names " +
                                        std::to_string(aux.size) +
                                        " banks, wants 1");
                        }
                        const std::string_view bank = plan.name(aux[0]);
                        const bool is_gate_up =
                            bank.find("gate_up") != std::string_view::npos;
                        // ISLAND (value arena). Every buffer is the
                        // statement's now: the block-major source is
                        // `inputs[0]`, the per-block expert id the kernel
                        // indexes the bank by is `inputs[1]`, and the
                        // destination is `inputs[2]` -- the staging the
                        // pointer build named, which the result aliases.
                        // The fork on the bank name picks WIDTHS and
                        // nothing else.
                        //
                        // `inputs[1]` used to be the sorted route order,
                        // which this kernel never reads: the statement
                        // named one array and the arm bound another
                        // (`mw.aligned_expert_ids`), so there was nothing
                        // the declaration could be checked against.
                        const auto gins = plan.inputs(op);
                        const auto gouts = plan.outputs(op);
                        need(gins, 3, "grouped gemm inputs");
                        need(gouts, 1, "grouped gemm outputs");
                        const int out_w = is_gate_up ? 2 * Im : H;
                        const int in_w = is_gate_up ? H : Im;
                        const auto* src = static_cast<const std::uint16_t*>(
                            values.slot(gins[0]));
                        const auto* expert_ids = static_cast<const std::int32_t*>(
                            values.slot(gins[1]));
                        auto* dst =
                            static_cast<std::uint16_t*>(values.slot(gouts[0]));
                        if (kernels::moe::moe_grouped_gemm_bf16_supported(
                                block, out_w, in_w)) {
                            kernels::moe::moe_grouped_gemm_bf16(
                                src, wb.require(bank).data(), dst, expert_ids,
                                max_blocks, block, out_w, in_w, stream);
                        } else {
                            // The batched-cuBLAS fallback the hand path
                            // takes when the grouped kernel refuses the
                            // shape; the pointer arrays are already built.
                            kernels::gemm::batched_act_x_wt_bf16(cublas.handle(),
                                reinterpret_cast<const void* const*>(
                                    is_gate_up ? mw.b_gu_ptrs.data()
                                               : mw.b_dn_ptrs.data()),
                                reinterpret_cast<const void* const*>(
                                    is_gate_up ? mw.a_gu_ptrs.data()
                                               : mw.a_dn_ptrs.data()),
                                reinterpret_cast<void* const*>(
                                    is_gate_up ? mw.c_gu_ptrs.data()
                                               : mw.c_dn_ptrs.data()),
                                block, out_w, in_w, max_blocks);
                        }
                        break;
                    }
                    case declared::Kernel::MoeReorderAligned: {
                        // ISLAND (value arena).
                        // `reorder_moe_aligned_output(down, sorted)` --
                        // both operands and the result are stated.
                        const auto rins = plan.inputs(op);
                        const auto routs = plan.outputs(op);
                        need(rins, 2, "reorder inputs");
                        need(routs, 1, "reorder outputs");
                        kernels::moe::reorder_moe_aligned_output_bf16(
                            values.slot(rins[0]),
                            static_cast<const std::int32_t*>(
                                values.slot(rins[1])),
                            values.slot(routs[0]), routes, aligned_rows, H,
                            shared_row_begin, N,
                            /*shared_out=*/nullptr, stream);
                        break;
                    }
                    // The COMBINE and the SHARED GATE generate. Both
                    // read their extents off the values -- the reorder
                    // above produces `[Tokens, top_k, hidden]`, so the
                    // route count and the width are its own dims -- and
                    // both accumulate into an operand the `kernel!` row
                    // aliases the result over. The shared gate's arm
                    // carried a warning that reversing its last two
                    // arguments lands the gate on the wrong buffer and
                    // still compiles; the row states that order once.
                    default:
                        break;
                    }
                }
                break;
            }
            }
            break;
        }
        case PieForwardOpKind::Guard: {
            // RUNG: the chain is resolved by `lower()`, which reads the
            // fire's rows and returns only the regions that run. A Guard
            // reaching an executor that drives the flat list means the
            // declaration and the drive disagree about who chooses.
            throw_drift("Guard op in a lowered drive");
            break;
        }
        case PieForwardOpKind::LmHead: {
            const std::string_view name = plan.weight_name(op);
            // Tied embeddings trace the lm head as "embed"; either way the
            // binding already aliased `w.lm_head` accordingly.
            const DeviceTensor* lm_head =
                name == "embed" ? &wb.require(name)
                : name == "lm_head" ? &wb.require(name)
                : nullptr;
            if (lm_head == nullptr) throw_unknown_weight(name);
            // The hand-written epilogue, copied whole: the final norm
            // already landed ALL rows in norm_x (the Rmsnorm arm above);
            // compact-logit fires gather the sampler rows into norm_y and
            // multiply just those, full emits multiply everything. Then
            // the full normed hidden is copied back to ws.y for MTP/state
            // plumbing — a fire-shape service the trace does not state,
            // exactly like the gather.
            // ISLAND (value arena). The normed hidden and the logits
            // are the statement's; `V` and `H` are their row widths.
            const auto lins = plan.inputs(op);
            const auto louts = plan.outputs(op);
            need(lins, 1, "lm_head inputs");
            need(louts, 1, "lm_head outputs");
            // SHARED ARM (D1): the compaction is identical in three
            // executors; only the head weight's resolution is not.
            int head_rows = N;
            const void* const head_in = declared::arm_epilogue_gather(
                {plan, values, N, 0, stream}, op, values.epilogue_gather(flat),
                logit_row_indices_d, num_logit_rows, &head_rows);
            kernels::gemm::act_x_w(cublas.handle(), head_in, *lm_head,
                                   values.slot(louts[0]), head_rows,
                                   row_width(louts[0]), row_width(lins[0]));
            // The copy-back is the OTHER fire-shape service the trace
            // does not state: MTP and the state plumbing read the full
            // normed hidden from `ws.y` after the epilogue. Its source
            // is the statement's operand; its destination is not a
            // traced value at this point in the fire.
            CUDA_CHECK(cudaMemcpyAsync(
                ws.y.data(), values.slot(lins[0]),
                static_cast<std::size_t>(N) * row_width(lins[0]) *
                    sizeof(std::uint16_t),
                cudaMemcpyDeviceToDevice, stream));
            break;
        }
        case PieForwardOpKind::HookSite: {
            // A4 + the 2026-08-05 ruling: qwen3_5's sites are
            // OBSERVATION-only and fire on FULL-ATTENTION layers only
            // (forward-hybrid.wit's contract); the observed buffer is the
            // roped q (bf16), the same the hand-written body exposes.
            if (stage_hooks == nullptr) break;
            const int L = static_cast<int>(op.param1);
            const StageHookPoint point = op.param0 == 0
                ? StageHookPoint::OnAttnProj
                : StageHookPoint::OnAttn;
            const bool full_attn =
                L >= 0 && L < static_cast<int>(w.layers.size()) &&
                w.layers[L].kind == LayerW::Kind::FullAttn;
            // forward-hybrid.wit ruling (2026-08-05): "the attention taps
            // fire on attention layers only" — a HookSite op on a GDN
            // layer is a no-op, and the hook ledger counts the
            // full-attention layers (context.cpp registers that count).
            if (full_attn) {
                // ISLAND (value arena). The observed buffer is the
                // seam's own value -- `attn.q` names the roped q -- so
                // the site stops naming `ws.q` and its width stops
                // being `Hq`.
                const auto hins = plan.inputs(op);
                need(hins, 1, "hook site inputs");
                invoke_stage_hook(
                    stage_hooks, point, values.slot(hins[0]),
                    static_cast<std::uint32_t>(N),
                    static_cast<std::uint32_t>(row_width(hins[0])),
                    static_cast<std::uint32_t>(L), stream);
            }
            break;
        }
        default:
            throw std::runtime_error(
                "declared qwen35 forward: op kind " +
                std::to_string(static_cast<std::uint32_t>(op.kind)) +
                " has no emission rule");
        }
    };

    // ── WHAT A DECLARED FIRE RUNS ──────────────────────────────────
    //
    // Build the fire's rows, lower them, execute the list — llama_like's
    // drive, at this family's much smaller vocabulary. Until this rung
    // there was a WALK here instead: the same switch, reached by a loop
    // that carried a guard-skip cursor and jumped dead regions itself.
    // The switch is untouched; what is gone is the traversal.
    //
    // Statements run in op order and both lists are in that order, so
    // this is a merge. Several rectangles can share one statement (an
    // arm that runs more than one kernel), and the arm runs them all
    // itself — so a statement is executed ONCE, at its first rectangle.
    std::size_t next_site = 0;
    std::size_t at = 0;
    while (at < flat.launches_len || next_site < flat.structural_len) {
        const bool site_first =
            at >= flat.launches_len ||
            (next_site < flat.structural_len &&
             flat.structural[next_site].at_op < flat.launches[at].at_op);
        if (site_first) {
            execute_op(plan.op(flat.structural[next_site].at_op),
                       flat.structural[next_site].at_op);
            ++next_site;
            continue;
        }
        const std::uint32_t at_op = flat.launches[at].at_op;
        while (at < flat.launches_len && flat.launches[at].at_op == at_op) {
            ++at;
        }
        execute_op(plan.op(at_op), at_op);
    }
    return true;
}

bool qwen3_5_forward_declared(
    const Qwen35DeclaredPlan& declared, const Qwen3_5Weights& w,
    const HfConfig& cfg, const Qwen3_5ForwardCfg& fwd_cfg,
    const Qwen3_5PlanState& plan_state, Workspace& ws,
    Qwen3_5MoeMlpWorkspace* moe_ws,
    Qwen3_5LinearAttnWorkspace& la, KvCache& cache,
    RecurrentStateCache& state_cache, AttentionWorkspace& attn_ws,
    kernels::gemm::CublasHandle& cublas,
    const std::int32_t* token_ids, const std::int32_t* positions,
    const std::uint32_t* qo_indptr, const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr, const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* qo_indptr_h, const std::uint32_t* kv_page_indptr_h,
    int total_tokens, int num_requests, bool is_pure_decode,
    const std::uint32_t* w_page_d, const std::uint32_t* w_off_d,
    const std::uint8_t* row_valid_d, bool has_write_desc,
    const std::int32_t* slot_ids_h, const std::uint8_t* is_fresh_h,
    const std::int32_t* slot_ids_d, const std::uint8_t* is_fresh_d,
    const std::int32_t* logit_row_indices_d, int num_logit_rows,
    const std::int32_t* commit_lens, const StageHooks* stage_hooks)
{
    return forward_declared_tmpl(
        declared, w, cfg, fwd_cfg, plan_state, ws, moe_ws, la, cache, state_cache,
        attn_ws, cublas, token_ids, positions, qo_indptr, kv_page_indices,
        kv_page_indptr, kv_last_page_lens, qo_indptr_h, kv_page_indptr_h,
        total_tokens, num_requests, is_pure_decode, w_page_d, w_off_d,
        row_valid_d, has_write_desc, slot_ids_h, is_fresh_h, slot_ids_d,
        is_fresh_d, logit_row_indices_d, num_logit_rows, commit_lens,
        stage_hooks);
}

bool qwen3_5_forward_declared(
    const Qwen35DeclaredPlan& declared, const Qwen3_5MoeWeights& w,
    const HfConfig& cfg, const Qwen3_5ForwardCfg& fwd_cfg,
    const Qwen3_5PlanState& plan_state, Workspace& ws,
    Qwen3_5MoeMlpWorkspace* moe_ws,
    Qwen3_5LinearAttnWorkspace& la, KvCache& cache,
    RecurrentStateCache& state_cache, AttentionWorkspace& attn_ws,
    kernels::gemm::CublasHandle& cublas,
    const std::int32_t* token_ids, const std::int32_t* positions,
    const std::uint32_t* qo_indptr, const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr, const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* qo_indptr_h, const std::uint32_t* kv_page_indptr_h,
    int total_tokens, int num_requests, bool is_pure_decode,
    const std::uint32_t* w_page_d, const std::uint32_t* w_off_d,
    const std::uint8_t* row_valid_d, bool has_write_desc,
    const std::int32_t* slot_ids_h, const std::uint8_t* is_fresh_h,
    const std::int32_t* slot_ids_d, const std::uint8_t* is_fresh_d,
    const std::int32_t* logit_row_indices_d, int num_logit_rows,
    const std::int32_t* commit_lens, const StageHooks* stage_hooks)
{
    return forward_declared_tmpl(
        declared, w, cfg, fwd_cfg, plan_state, ws, moe_ws, la, cache, state_cache,
        attn_ws, cublas, token_ids, positions, qo_indptr, kv_page_indices,
        kv_page_indptr, kv_last_page_lens, qo_indptr_h, kv_page_indptr_h,
        total_tokens, num_requests, is_pure_decode, w_page_d, w_off_d,
        row_valid_d, has_write_desc, slot_ids_h, is_fresh_h, slot_ids_d,
        is_fresh_d, logit_row_indices_d, num_logit_rows, commit_lens,
        stage_hooks);
}

}  // namespace pie_cuda_driver::model
