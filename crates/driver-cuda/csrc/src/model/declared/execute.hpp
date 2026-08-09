#pragma once

// THE SHARED SWITCH — one arm per symbol, for every symbol whose
// execution is the same wherever it is stated.
//
// There were four switches, one per family executor, and a census of
// their cases found most of each already family-blind: 13 of
// llama_like's 24, 17 of gemma-4's, 17 of gpt-oss's. "Family-blind"
// means the body mentions no workspace field, no weights struct and no
// config — what is left is the statement's operands, the rectangle's
// rows, and the fire's own inputs.
//
// Those are not four arms. They are one arm written four times, and
// where they differed they differed by ACCIDENT rather than by family:
// llama_like's `WriteKvToPages` takes a device-window form under
// hook-graph capture and gemma-4's does not, which is a property of the
// FIRE (is there a peel?) and not of the model. So the window joins the
// context and one body serves both.
//
// [`execute_shared`] returns false for a symbol it does not own, and the
// caller's own switch runs. That residue is what is left of a family
// executor, and it shrinks as arms land here — which is the measure
// this file exists to make, rather than a claim it exists to support.
//
// ── WHERE AN ARM COMES FROM, and it is not this file ────────────────
//
// "A new kernel means one more case here" is the answer this design
// exists to avoid. The answer it gives instead: the `kernel!` ROW says
// where each argument comes from, and the arm is DERIVED —
// `generated_dispatch.inc`, included below, one branch per fully-stated
// row. A hand-written case here is the fallback, not the route.
//
// Two things that only became visible once arms started generating, and
// both are the same shape — a REFUSAL that lived in an arm rather than
// in the row it was about:
//
//   * The rope arms refuse a zero theta, because gemma-4 alternates
//     theta per layer and says so by leaving the context field zero.
//     The generated branch inherited the argument and not the refusal,
//     and it runs FIRST — so it would have rotated half that model by
//     nothing. `Source::CtxNonZero` moves the refusal into the row.
//   * `Source::Rows` binds the FIRE's row count, which is not every
//     statement's. The MoE aligned leg's rows are a padded block-major
//     count, and a branch handing that kernel `N` activates the first N
//     of them. `Source::OutRows` resolves the value's own extent, which
//     is both the fix and the deletion of a formula five hand-written
//     forwards restate.
//
// The rule they teach: a fact an arm KNOWS is a fact the row should
// SAY, because an arm can be replaced by a generated one and a row
// cannot.

#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "attention_workspace.hpp"
#include "attn/attention_flashinfer.hpp"
#include "attn/attn_sink.hpp"
#include "attn/head_dim_pad.hpp"
#include "attn/kv_paged.hpp"
#include "gemm/gemm.hpp"
#include "layout/deinterleave.hpp"
#include "layout/gather_rows.hpp"
#include "norm/residual_add.hpp"
#include "norm/rmsnorm.hpp"
// Headers the GENERATED half calls into. A row that states its sources
// puts a call here, and the launcher has to be declared for it — which
// is how this list grows and why it is not a curated one: the compiler
// names the header the moment a row starts generating without it.
#include "norm/altup_aux.hpp"
#include "norm/dsv4_hc.hpp"
#include "norm/scalar_mul.hpp"
#include "ssm/gated_delta_net.hpp"
#include "mlp/swiglu.hpp"
#include "moe/moe_dispatch.hpp"
#include "moe/topk_softmax.hpp"
#include "quant/dequant_wna16.hpp"
#include "rope/rope.hpp"
#include "distributed.hpp"
#include "store/recurrent_state_cache.hpp"
#include "store/kv_cache.hpp"
#include "model/declared/arms.hpp"
#include "model/declared/registry.hpp"
#include "model/declared/value_arena.hpp"
#include "model/declared/weights.hpp"

namespace pie_cuda_driver::model::declared {

// WHAT A SHARED ARM MAY READ, and nothing more.
//
// [`ArmCtx`] is the inner half — the plan, the arena, the rectangle.
// This adds what a LAUNCH needs beyond an arm: the fire's own inputs,
// the binder, and the handles a kernel is given rather than told.
//
// The head geometry is here for the reason `eps` is a parameter to
// `arm_rmsnorm`: it is config the trace does not carry, and a row width
// divided by a head count is a carving only once you already know one
// of them. What is NOT here is anything a family knows and another
// does not — the moment a field would be one family's, the arm it
// serves is not shared and belongs in that family's residue.
struct ExecCtx {
    // The inner half, verbatim.
    ArmCtx arm;

    const WeightBinder& wb;
    KvCache& cache;
    AttentionWorkspace& attn_ws;
    kernels::gemm::CublasHandle& cublas;
    // The rank's communicator, or null on a single-GPU deployment. A
    // HANDLE like `cublas` and `cache`: the arms are given it, never
    // told about it — which is the difference between a collective
    // being a statement and being a side effect reached for through
    // `tp->` from inside a body.
    NcclComm* tp_comm = nullptr;
    // The RECURRENT STATE store, for the families that have one. A
    // handle like the KV cache: qwen3.5's GDN arms address per-request
    // conv and recurrence slots through it, and a slot is not a traced
    // value.
    RecurrentStateCache* state_cache = nullptr;

    // The fire's inputs.
    const std::int32_t* positions = nullptr;
    const std::uint32_t* qo_indptr = nullptr;
    const std::uint32_t* kv_page_indices = nullptr;
    const std::uint32_t* kv_page_indptr = nullptr;
    const std::uint32_t* kv_last_page_lens = nullptr;
    const std::uint8_t* row_valid = nullptr;
    // The HOST indptrs the plan-free prefill wrapper builds its own
    // R-shaped plan from on the way in.
    const std::uint32_t* qo_indptr_h = nullptr;
    const std::uint32_t* kv_page_indptr_h = nullptr;
    // The explicit KV write's descriptors; null on a page-derived fire.
    const std::uint32_t* w_page_d = nullptr;
    const std::uint32_t* w_off_d = nullptr;
    int num_requests = 0;

    // The PEEL's device window, and which face this launch serves.
    // Null and false are the plain form, which is why one body covers
    // both: whether a fire has a peel is the fire's property.
    const std::uint32_t* peel_window_d = nullptr;
    bool peel_tail = false;

    // Config the trace does not carry.
    float eps = 0.f;
    // The softmax scale a dispatch runs at, or -1 to let the kernel take
    // `1/sqrt(head_dim)`. gemma-4 overrides it (its query is pre-scaled),
    // and a padded deployment overrides it to the LOGICAL dim.
    float sm_scale = -1.f;
    // Where a dispatch writes its LSE when the statement declares no
    // second result. gpt-oss is the only family that declares one.
    float* lse_fallback = nullptr;
    float rope_theta = 0.f;
    int num_q_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    // The width the attention kernels run at; equal to `head_dim` where
    // nothing pads.
    int head_dim_kernel = 0;

    // The CACHE SLOT this launch's KV write addresses — already
    // resolved, not a model layer index.
    //
    // The two are not the same everywhere and assuming they were is a
    // defect this merge shipped for one commit: qwen3.5 gives only its
    // FULL-ATTENTION layers a cache slot, so its model layer 7 may be
    // cache layer 1. The mapping is the family's (the trace does not
    // state it), so the family resolves it and hands the answer over.
    //
    // Negative means this fire's layer has no cache slot, which is a
    // drift if a KV write is stated over it.
    int kv_layer = -1;

    // ── THE ATTENTION PLAN, already resolved ────────────────────────
    //
    // A flashinfer dispatch takes a PLAN the prepare built, and which
    // plan is the one thing about the dispatch that is not the
    // statement's. Every family had its own answer and therefore its
    // own copy of one call: llama_like and qwen3.5 read a `plan_state`,
    // mixtral builds one per fire, gemma-4 keeps TWO and picks by
    // whether the layer is full-attention or sliding — and llama_like
    // swaps in a DEPTH-PREFIX plan for one launch per layer of a union
    // tail.
    //
    // None of that is derivable here, and all of it is derivable
    // there. So this follows `kv_layer`: the family resolves and hands
    // the answer over, per op if it varies per op, and the arm takes a
    // plan the way it takes `cublas` — given, never reached for.
    //
    // Null means "this fire built none", which is a drift if a dispatch
    // is stated over it, and the arm says so by name.
    const kernels::attn::DecodePlanCache* decode_plan = nullptr;
    const kernels::attn::PrefillPlanCache* prefill_plan = nullptr;

    // Where a launch writes when the statement declares no result.
    //
    // The GUARD-REGION spelling, and it is not the attention's alone: a
    // value-producing guard owns the output and its arms record none,
    // so the destination is the guard's and only the family's walk
    // knows which value that is. qwen3.5's recurrence three-way is the
    // same shape as gpt-oss's attention chain.
    //
    // `lse_fallback`'s peer, and null is the honest default — a launch
    // handed neither a result nor a region destination refuses rather
    // than writing somewhere plausible.
    void* region_dst = nullptr;

    // ── THE RECURRENT STATE, already resolved ──────────────────────
    //
    // `state_cache` above is the STORE; these are this launch's window
    // into it, and resolving them is the family's the way `kv_layer` is.
    // The slab is `recurrent_state_raw(layer, 0)` for the layer the
    // statement marks, and null where the statement marks none — which
    // is a drift if a recurrence is stated over it.
    //
    // `slot_ids` and `slot_stride` address a REQUEST's slab inside it;
    // `write_state` is the frozen-verify pass suppressing its own
    // writes; `commit_lens` is the spec-decode repair's per-request
    // length, null on an ordinary fire. All four are the FIRE's, not
    // the statement's, which is why they ride here and not in the param
    // channel.
    void* rs_slab = nullptr;
    const std::int32_t* rs_slot_ids = nullptr;
    long long rs_slot_stride = 0;
    bool write_state = true;
    const std::int32_t* commit_lens = nullptr;
};

// Run `op`'s arm if this file owns its symbol.
//
// Returns false when the symbol needs the family's own half — which is
// an answer, not a failure: `resolve_kernel` already refused anything
// the registry does not know, so a false here means "stated, and this
// family executes it its own way".
inline bool execute_shared(const ExecCtx& c,
                           const pie_forward::PieForwardOp& op) {
    const auto& plan = c.arm.plan;
    auto& values = c.arm.values;
    const int N = c.arm.rows;
    const auto stream = c.arm.stream;
    const auto aux = plan.aux_names(op);
    const auto ins = plan.inputs(op);
    const auto outs = plan.outputs(op);

    // ── THE GENERATED HALF ─────────────────────────────────────────
    //
    // A row that states both its operand TYPES and where each argument
    // COMES FROM is a row this call can be derived from, and
    // `generated_dispatch.inc` is that derivation — one branch per such
    // row, regenerated by `cargo run -p kernels-cuda --bin
    // emit-dispatch`.
    //
    // It runs FIRST, and everything below it is the residue: arms whose
    // rows have not stated their sources, or cannot. `rope_partial`'s
    // Q-only spelling is the second kind — it states one result and
    // reaches the launcher with `num_kv_heads = 0`, which is arity the
    // STATEMENT carries and not a source a row can name.
    //
    // The generated branches read `sym`, `ins`, `outs`, `aux` and `ps`;
    // they are bound once here because every branch wants them.
    const auto ps = plan.aux_params(op);
    const std::string_view sym = plan.weight_name(op);
    (void)ps;
    // WHERE A REGION LAUNCH WRITES. The statement's result where it
    // declares one, the enclosing guard's value otherwise — see
    // `Source::ResultOrRegion`. A launch that has neither is a drift,
    // and refusing is the only safe answer: writing somewhere plausible
    // is how a guard chain silently produces the wrong region's output.
    const auto result_or_region = [&](const ExecCtx& cc,
                                      const pie_forward::ForwardPlan::IdSpan& o,
                                      std::size_t i) -> void* {
        if (o.size > i) return cc.arm.row(o[i]);
        if (cc.region_dst != nullptr) return cc.region_dst;
        throw std::runtime_error(
            "declared arm: a launch declares no result and sits in no "
            "value-producing guard, so it has nowhere to write");
    };
    (void)result_or_region;
#include "model/declared/generated_dispatch.inc"

    // One weight, required, by the name the statement gave.
    const auto one_weight = [&](const char* what) -> const DeviceTensor& {
        if (aux.size != 1) {
            throw std::runtime_error(
                std::string("declared arm: a stated ") + what + " names " +
                std::to_string(aux.size) + " weights, wants 1");
        }
        return c.wb.require(plan.name(aux[0]));
    };

    switch (resolve_kernel(plan.weight_name(op))) {
    // The row norms, the residual landing, both swiglu spellings, the
    // fused residual norm, the router, the cast and the routed combine
    // are GENERATED now -- their rows state their sources, so the
    // branch above bound them and this switch never sees them. What is
    // left below is what a row cannot yet say.


    case Kernel::RopeFull:
    case Kernel::RopePartial: {
        // A caller whose theta varies per layer states so by leaving
        // `rope_theta` zero, and keeps its own arm. Refusing here is
        // what makes that safe: a zero theta rotates by nothing, and
        // silently.
        if (c.rope_theta == 0.f) return false;
        need(outs, 1, "rope outputs");
        const bool q_only = outs.size < 2;
        void* const rq = c.arm.row(outs[0]);
        void* const rk = q_only ? rq : c.arm.row(outs[1]);
        const int kv_heads = q_only ? 0 : c.num_kv_heads;
        // `[rotary_dim]`, zero for the full rotation.
        //
        // The THETA is the CONTEXT's and not the statement's, and that
        // is the one thing this arm still reads that it should not:
        // gemma-4 alternates theta per layer, so a caller handing one
        // config value is handing the wrong one for half that model.
        // gemma-4 therefore keeps its own rope arm; see `dsl::cuda::rope`
        // for what unblocks moving it here.
        const auto ps = plan.aux_params(op);
        if (ps.size != 1) {
            throw std::runtime_error(
                "declared arm: a rotation states " +
                std::to_string(ps.size) +
                " scalar arguments, wants 1 (rotary_dim)");
        }
        if (resolve_kernel(plan.weight_name(op)) == Kernel::RopePartial) {
            kernels::rope::rope_partial_bf16(
                rq, rk, c.positions, N, c.num_q_heads, kv_heads, c.head_dim,
                static_cast<int>(ps[0]), c.rope_theta, stream);
        } else {
            kernels::rope::rope_bf16(
                rq, rk, c.positions, N, c.num_q_heads, kv_heads, c.head_dim,
                c.rope_theta, stream);
        }
        return true;
    }

    // ── the head-dim staging ───────────────────────────────────────
    case Kernel::PadHeadDim:
    case Kernel::StripHeadDim: {
        need(ins, 1, "head-dim staging inputs");
        need(outs, 1, "head-dim staging outputs");
        const auto& rv = plan.value(outs[0]);
        if (rv.rank != 3) {
            throw std::runtime_error(
                "declared arm: a head-dim staging result states rank " +
                std::to_string(rv.rank) + ", wants [Tokens, heads, dim]");
        }
        const int heads = static_cast<int>(rv.dims[1].value);
        if (resolve_kernel(plan.weight_name(op)) == Kernel::PadHeadDim) {
            kernels::attn::pad_head_dim_bf16(
                c.arm.row(ins[0]), c.arm.row(outs[0]), N, heads,
                c.head_dim, c.head_dim_kernel, stream);
        } else {
            kernels::attn::strip_head_dim_bf16(
                c.arm.row(ins[0]), c.arm.row(outs[0]), N, heads,
                c.head_dim, c.head_dim_kernel, stream);
        }
        return true;
    }

    // ── the KV write ───────────────────────────────────────────────
    //
    // The device-window form is the PEEL's, and a peel is a property of
    // the fire: llama_like's hook-graph captures carry one and gemma-4's
    // fires do not, which is why one body serves both and the fork reads
    // the context rather than the family.
    // The EXPLICIT write: the fire steered a graph replay, so it
    // carries per-row page/offset descriptors instead of deriving them.
    // Its device-window form is the peel's, exactly as the paged one's
    // is.
    case Kernel::WriteKvExplicit: {
        need(ins, 2, "kv write inputs");
        if (c.w_page_d == nullptr || c.w_off_d == nullptr) {
            throw std::runtime_error(
                "declared arm: an explicit KV write is stated but the fire "
                "carries no descriptors");
        }
        if (c.kv_layer < 0) {
            throw std::runtime_error(
                "declared arm: a KV write is stated over a layer with no "
                "cache slot");
        }
        auto kv_view = c.cache.layer_view(c.kv_layer);
        if (c.peel_window_d != nullptr && c.peel_tail) {
            kernels::attn::write_kv_explicit_bf16_devwin(
                kv_view, c.arm.row(ins[0]), c.arm.row(ins[1]),
                c.w_page_d, c.w_off_d, c.peel_window_d, N, stream,
                c.row_valid);
            return true;
        }
        const int lo = c.arm.win_start;
        kernels::attn::write_kv_explicit_bf16(
            kv_view,
            static_cast<const std::uint16_t*>(c.arm.row(ins[0])) +
                static_cast<std::size_t>(lo) * row_width(plan, ins[0]),
            static_cast<const std::uint16_t*>(c.arm.row(ins[1])) +
                static_cast<std::size_t>(lo) * row_width(plan, ins[1]),
            c.w_page_d + lo, c.w_off_d + lo, N, stream,
            c.row_valid != nullptr ? c.row_valid + lo : nullptr);
        return true;
    }

    case Kernel::WriteKvToPages: {
        need(ins, 2, "kv write inputs");
        if (c.kv_layer < 0) {
            throw std::runtime_error(
                "declared arm: a KV write is stated over a layer with no "
                "cache slot");
        }
        auto kv_view = c.cache.layer_view(c.kv_layer);
        if (c.peel_window_d != nullptr && c.peel_tail) {
            // THE BASE, not the window. A device-window form reads its
            // split off `peel_window_d` and addresses the fire's rows
            // itself, so advancing the pointer would apply the offset
            // twice. That is the one place `row()` is the wrong answer,
            // and it is a property of the CALL FORM — which is why these
            // stay hand-written arms.
            kernels::attn::write_kv_to_pages_bf16_devwin(
                kv_view, values.slot(ins[0]), values.slot(ins[1]),
                c.qo_indptr, c.kv_page_indices, c.kv_page_indptr,
                c.kv_last_page_lens, c.peel_window_d, N, c.num_requests,
                stream, c.row_valid);
            return true;
        }
        kernels::attn::write_kv_to_pages(
            kv_view, c.arm.row(ins[0]), c.arm.row(ins[1]), c.qo_indptr,
            c.kv_page_indices, c.kv_page_indptr, c.kv_last_page_lens, N,
            c.num_requests, stream, c.row_valid, /*first_token=*/0);
        return true;
    }

    // ── the fused q/k norm + rotation ──────────────────────────────
    //
    // ONE stated symbol, two launchers, and the fork is the FIRE's: a
    // hook peel's tail region carries a device word and the plain form
    // does not. `WriteKvToPages` above is the same shape, and the reason
    // this is shared rather than llama_like's is the same too — whether
    // a fire has a peel is not a property of the model.
    //
    // A caller whose theta varies per layer keeps its own arm, by
    // leaving `rope_theta` zero; see `Source::CtxNonZero`.
    case Kernel::QkRmsnormRope: {
        if (c.rope_theta == 0.f) return false;
        need(outs, 2, "fused qk-norm+rope outputs");
        need(aux, 2, "fused qk-norm+rope weights");
        const void* const q_w = c.wb.require(plan.name(aux[0])).data();
        const void* const k_w = c.wb.require(plan.name(aux[1])).data();
        if (c.peel_window_d != nullptr && c.peel_tail) {
            kernels::rope::qk_rmsnorm_rope_bf16_devwin(
                values.slot(outs[0]), values.slot(outs[1]), q_w, k_w,
                c.positions, c.peel_window_d, N, c.num_q_heads,
                c.num_kv_heads, c.head_dim, c.rope_theta, c.eps, stream);
            return true;
        }
        kernels::rope::qk_rmsnorm_rope_bf16(
            c.arm.row(outs[0]), c.arm.row(outs[1]), q_w, k_w,
            c.positions + c.arm.win_start, N, c.num_q_heads,
            c.num_kv_heads, c.head_dim, c.rope_theta, c.eps, stream);
        return true;
    }

    case Kernel::AllReduce:
    case Kernel::AllReduceOut: {
        if (c.tp_comm == nullptr) {
            throw std::runtime_error(
                "declared arm: the trace states a collective but this "
                "deployment bound no communicator (tp_size and tp_comm "
                "disagree)");
        }
        need(ins, 1, "all-reduce inputs");
        need(outs, 1, "all-reduce outputs");
        c.tp_comm->all_reduce_bf16_out(
            c.arm.row(ins[0]), c.arm.row(outs[0]),
            static_cast<std::size_t>(N) *
                static_cast<std::size_t>(row_width(plan, outs[0])),
            ncclSum, stream);
        return true;
    }

    // The two-step landing's second half: `y += summed` and the norm of
    // the sum, one launch. Operand 0 is the residual stream (updated in
    // place), operand 1 the summed partial, and the result is the
    // normed activation the MLP reads.

    case Kernel::MatmulTensorScaled:
    case Kernel::MatmulChannelScaled:
    case Kernel::MatmulGroupedScaled:
    case Kernel::MatmulMxfp4Marlin: {
        if (aux.size < 2 || aux.size > 3) {
            throw std::runtime_error(
                "declared arm: a scaled projection names " +
                std::to_string(aux.size) +
                " weights, wants 2 (W, scales) or 3 (+ zeros)");
        }
        const auto matched = resolve_kernel(plan.weight_name(op));
        const ScaledRepr repr =
            matched == Kernel::MatmulTensorScaled  ? ScaledRepr::PerTensor
            : matched == Kernel::MatmulChannelScaled ? ScaledRepr::PerChannel
            : matched == Kernel::MatmulGroupedScaled ? ScaledRepr::PerGroup
                                                     : ScaledRepr::Mxfp4Marlin;
        arm_scaled_matmul(c.arm, op, repr, c.cublas.handle(),
                          c.wb.require(plan.name(aux[0])),
                          c.wb.require(plan.name(aux[1])),
                          aux.size == 3 ? &c.wb.require(plan.name(aux[2]))
                                        : nullptr,
                          // A quantized projection never folds its
                          // residual: `try_fold_residual` refuses a
                          // `Launch`, so the landing is a stated add.
                          0.f);
        return true;
    }

    // ── the plan-free prefill dispatch ─────────────────────────────
    //
    // A DIFFERENT statement from the planned one, not a spelling of it:
    // this wrapper builds its own R-shaped plan from the host indptrs on
    // the way in, which is why it needs no plan cache and why it is the
    // one attention dispatch that can be shared outright.
    //
    // gemma-4 and gpt-oss both state it and differed in two arguments,
    // both the CALLER's: the softmax scale (gemma-4 pre-scales its
    // query, so it passes 1.0) and where the LSE lands.
    case Kernel::AttnFlashinferPrefillPlanless: {
        need(ins, 1, "prefill attention inputs");
        need(outs, 1, "prefill attention outputs");
        if (c.kv_layer < 0) {
            throw std::runtime_error(
                "declared arm: a prefill dispatch is stated over a layer "
                "with no cache slot");
        }
        kernels::attn::attention_flashinfer_prefill(
            c.arm.row(ins[0]), c.cache.layer_view(c.kv_layer),
            c.arm.row(outs[0]), c.qo_indptr, c.kv_page_indices,
            c.kv_page_indptr, c.kv_last_page_lens, c.qo_indptr_h,
            c.kv_page_indptr_h, N, c.num_requests, c.num_q_heads,
            c.attn_ws.view(), stream, stated_window_left(plan, op),
            /*logits_soft_cap=*/0.f, c.sm_scale,
            outs.size >= 2 ? static_cast<float*>(c.arm.row(outs[1]))
                           : c.lse_fallback);
        return true;
    }

    // ── the PLANNED dispatches ─────────────────────────────────────
    //
    // One arm each, for the first time. These were four copies of one
    // call: what differed between them was the PLAN, and the plan is
    // now the context's — resolved by the family, which is the only
    // party that can (see `decode_plan`).
    //
    // What survives as an argument rather than a fork: the window
    // (stated), the softmax scale (the context's, because a pre-scaled
    // query and a padded head dim are both the caller's business), and
    // the destination when the statement declares none.
    case Kernel::AttnFlashinferDecode: {
        if (c.decode_plan == nullptr) return false;
        if (c.kv_layer < 0) {
            throw std::runtime_error(
                "declared arm: a decode dispatch is stated over a layer "
                "with no cache slot");
        }
        arm_attention_decode(c.arm, op, *c.decode_plan,
                             c.cache.layer_view(c.kv_layer),
                             c.kv_page_indices, c.kv_page_indptr,
                             c.kv_last_page_lens, c.attn_ws.view(),
                             stated_window_left(plan, op), c.sm_scale,
                             c.lse_fallback, c.region_dst);
        return true;
    }

    case Kernel::AttnFlashinferPrefill: {
        if (c.prefill_plan == nullptr) return false;
        if (c.kv_layer < 0) {
            throw std::runtime_error(
                "declared arm: a prefill dispatch is stated over a layer "
                "with no cache slot");
        }
        need(ins, 1, "prefill attention inputs");
        // The guard-region spelling again: the arms of a value-producing
        // guard record no result, so the destination is the guard's.
        void* const dst = outs.size > 0 ? c.arm.row(outs[0])
                                        : c.region_dst;
        if (dst == nullptr) {
            throw std::runtime_error(
                "declared arm: a prefill dispatch names no destination");
        }
        const auto kv_view = c.cache.layer_view(c.kv_layer);
        // No `window_left` here, unlike the decode: this launcher takes
        // none. A windowed deployment's prefill carries its window in
        // the PLAN the prepare built, which is one more reason the plan
        // belongs to the family rather than to this call.
        kernels::attn::dispatch_attention_flashinfer_prefill_bf16(
            *c.prefill_plan, c.arm.row(ins[0]), kv_view.k_bf16_pages,
            kv_view.v_bf16_pages, dst, c.qo_indptr, c.kv_page_indices,
            c.kv_page_indptr, c.kv_last_page_lens, c.attn_ws.view(), stream,
            /*logits_soft_cap=*/0.f, c.sm_scale,
            outs.size >= 2 ? static_cast<float*>(c.arm.row(outs[1]))
                           : c.lse_fallback);
        return true;
    }

    default:
        return false;
    }
}

}  // namespace pie_cuda_driver::model::declared
