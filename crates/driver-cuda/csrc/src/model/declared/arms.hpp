#pragma once

// SHARED OP ARMS — the executor's body, for the ops whose execution is
// already family-blind.
//
// The audit that started this merge found 13 of 23 op kinds present in
// both family executors, with the bodies differing only by OPERAND
// CONVENTION (which workspace buffer plays each role) and by the weights
// struct — never by arithmetic. Step 1 removed the weights difference (a
// binder); step 2 removed the walk's; this file is where an arm lands the
// moment its operands stop being a family's private convention.
//
// It starts with the arms that were ALREADY identical, character for
// character, in both executors — the strongest possible evidence that the
// executor wanted to be one file. The rest follow as the SSA value arena
// (the trace already carries `inputs`/`outputs`; what it does not carry is
// a buffer, because buffer assignment is a backend job that was written as
// family convention) replaces the routing conventions.

#include <bit>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "attn/attention_flashinfer.hpp"
#include "attn/attention_mla.hpp"
#include "moe/moe_dispatch.hpp"
#include "moe/topk_sigmoid.hpp"
#include "attn/mla_paged.hpp"
#include "attn/split_packed.hpp"
#include "gemm/gemm.hpp"
#include "layout/embed.hpp"
#include "layout/gather_rows.hpp"
#include "norm/residual_add.hpp"
#include "norm/rmsnorm.hpp"
#include "mlp/swiglu.hpp"
#include "model/declared/value_arena.hpp"
#include "model/workspace.hpp"

namespace pie_cuda_driver::model::declared {

// `Swiglu`: the packed-bank form when the MLP's gate/up matmul landed in
// the fused bank, the two-buffer form otherwise. `gate_up_used_fused` is
// the Matmul arm's own decision, carried forward — the trace states ONE
// packed matmul either way (see the binder's `gate_up`).
//
// Both executors held this arm character-for-character identical.
// `dst` is the traced value's slot once the caller has moved this island
// onto the arena; a caller that has not keeps passing its convention.
inline void arm_swiglu(Workspace& ws,
                       bool gate_up_used_fused,
                       void* dst,
                       int n,
                       int intermediate,
                       cudaStream_t stream) {
    if (gate_up_used_fused) {
        kernels::mlp::chunked_swiglu_bf16(
            ws.gate_up_fused.data(), dst, n, intermediate, stream);
    } else {
        kernels::mlp::swiglu_bf16(
            ws.gate.data(), ws.up.data(), dst, n * intermediate, stream);
    }
}

// ── the arms that read their operands off the plan ─────────────────
//
// D1's shape, one arm at a time. An arm lands here the moment its body
// stops mentioning a workspace field: what is left is the statement's
// operands, its widths, and the fire's row count, none of which is a
// family's to know.
//
// The two guards below travel with them, because both were written per
// executor and both catch real defects. `need` refuses a short operand
// span -- indexing past one does not fault, it reads the NEXT
// statement's operands and hands the arm a plausible pointer to the
// wrong buffer. `row_width` is a value's trailing dims, which is what
// `Hq`, `Hk`, `I`, `H` and the rest were spelling.

// WHAT EVERY ARM TAKES, and nothing more.
//
// The five arms below had converged on the same first four parameters —
// the plan to read the statement from, the arena to resolve its
// operands in, the rectangle's row count, and the stream. That is the
// shape D1 is heading for: one context, the statement, and whatever the
// FAMILY has to add.
//
// `win_start` is here rather than in the one arm that reads it because
// it is the same kind of fact as `rows`: both describe the RECTANGLE a
// launch covers, and a driver that walked rectangles rather than ops
// would hand the pair to every arm without asking which cares.
//
// What is NOT here is the family's half — a weight pointer, `eps`, a
// vocabulary size, a quantization descriptor. Those stay explicit
// arguments, one per arm, because making them a bag would hide which
// arm needs which, and the whole point of the exercise is that the list
// gets shorter as the trace states more.
inline void need(const pie_forward::ForwardPlan::IdSpan& span,
                 std::size_t n, const char* what) {
    if (span.size < n) {
        throw std::runtime_error(
            std::string("declared arm: ") + what + " states " +
            std::to_string(span.size) + " operands, needs " +
            std::to_string(n));
    }
}

inline int row_width(const pie_forward::ForwardPlan& plan,
                     std::uint32_t id) {
    const auto& val = plan.value(id);
    std::uint32_t out = 1;
    for (std::uint32_t d = 1; d < val.rank; ++d) {
        if (val.dims[d].kind != pie_forward::PieForwardDimKind::Const) {
            return 0;
        }
        out *= val.dims[d].value;
    }
    return static_cast<int>(out);
}

struct ArmCtx {
    const pie_forward::ForwardPlan& plan;
    ValueArena& values;
    /// Rows of the rectangle this launch covers.
    int rows;
    /// Its first row, in the fire's row space. Zero for the plain form.
    int win_start;
    cudaStream_t stream;

    /// WHERE THIS RECTANGLE'S SLICE OF A VALUE BEGINS.
    ///
    /// `c.row(id)` is the value's BASE, which is the whole of it
    /// only when the rectangle is the whole fire. A hook peel splits a
    /// layer body into two rectangles and the second one starts at
    /// `win_start`, so a launch over it writes rows `[win_start,
    /// win_start + rows)` — at the base it would write the first `rows`
    /// instead, which is the prefix region's rows, with the tail
    /// region's positions and weights.
    ///
    /// llama_like's own arms did this with a family helper (`bf16_row`);
    /// the shared ones did not, and neither did the generated branches,
    /// which is what makes this a member rather than a helper anyone may
    /// forget. `ArmCtx`'s own note predicted it: `win_start` "is the
    /// same kind of fact as `rows`: both describe the RECTANGLE a launch
    /// covers, and a driver that walked rectangles rather than ops would
    /// hand the pair to every arm without asking which cares."
    ///
    /// Only a TOKEN-rowed value has rows to window. A `[Requests, ...]`
    /// or block-major value is not sliced by a token range, and
    /// offsetting one would be the same defect in the other direction.
    void* row(std::uint32_t id) const {
        void* const base = values.slot(id);
        if (win_start == 0) return base;
        const auto& val = plan.value(id);
        if (val.rank == 0 ||
            val.dims[0].kind != pie_forward::PieForwardDimKind::Tokens) {
            return base;
        }
        const std::size_t width =
            static_cast<std::size_t>(row_width(plan, id));
        const std::size_t elem =
            (val.dtype == pie_forward::PieForwardDType::F32 ||
             val.dtype == pie_forward::PieForwardDType::I32)
                ? 4u
                : 2u;
        return static_cast<std::uint8_t*>(base) +
               static_cast<std::size_t>(win_start) * width * elem;
    }
};

// A FLOAT out of the param channel.
//
// `aux_params` is untyped `u32` — "what each slot means is the SYMBOL's
// contract" — so a statement carrying a scale carries its BITS. This is
// the read; `std::bit_cast` rather than a union or a reinterpret, so it
// is one constant-folded move and not a strict-aliasing question.
inline float f32_param(std::uint32_t bits) {
    return std::bit_cast<float>(bits);
}

// A VALUE'S LEADING EXTENT, resolved for this fire.
//
// `row_width` above answers "how wide is a row of this"; this answers
// "how many rows", and the two together are what a row-shaped launcher
// takes. Most values answer `Tokens` and the rectangle's count is
// already the answer — which is why `Source::Rows` existed first and
// covers most rows.
//
// The case it does NOT cover is the MoE aligned leg. Its values are
// `Dim::MoeAlignedRoutes`: every (token, expert) route bucketed by
// expert and each bucket padded to a whole block, so the extent is
// `ceil((N·k + min(E, N·k)·(block-1)) / block) · block` — a function of
// the fire's token count and three load-time numbers, and the one
// extent in the tree that is neither the fire's rows nor a constant.
// The dim PACKS those three into its `value` word (see
// `PieForwardDimKind::MoeAlignedRoutes`), so this is the whole of what
// it takes to resolve, and it is the only place in the driver that
// does — five hand-written forwards restate the formula, and each
// restatement is a place it can drift.
inline int value_rows(const pie_forward::ForwardPlan& plan,
                      std::uint32_t id, int fire_rows, int fire_requests) {
    const auto& val = plan.value(id);
    if (val.rank == 0) return fire_rows;
    const auto& d0 = val.dims[0];
    switch (d0.kind) {
    case pie_forward::PieForwardDimKind::Tokens:
        return fire_rows;
    case pie_forward::PieForwardDimKind::Requests:
        return fire_requests;
    case pie_forward::PieForwardDimKind::Const:
        return static_cast<int>(d0.value);
    case pie_forward::PieForwardDimKind::MoeAlignedRoutes: {
        const int block = static_cast<int>(d0.value & 0xffu);
        const int experts = static_cast<int>((d0.value >> 8) & 0xffffu);
        const int top_k = static_cast<int>((d0.value >> 24) & 0xffu);
        if (block <= 0) {
            throw std::runtime_error(
                "declared arm: an aligned extent packs a zero block size");
        }
        const int routes = fire_rows * top_k;
        const int cap = std::min(experts, routes);
        return ((routes + cap * (block - 1) + block - 1) / block) * block;
    }
    }
    return fire_rows;
}

// THIS LAYER'S SLIDING WINDOW, off the statement.
//
// Every attention dispatch states it (`dsl::cuda::attn_at`'s params),
// because it is a load-time fact -- a config's `sliding_window`, or its
// per-layer list where the architecture alternates -- and a load-time
// fact is a trace-time constant. What stood here instead was three
// lines reaching into `fwd_cfg.per_layer_window_left`, an array no
// statement mentioned, at every dispatch in four executors.
//
// Signed through the unsigned channel: `-1` arrives as `0xFFFFFFFF` and
// casts back, which is the params channel's stated convention.
inline int stated_window_left(const pie_forward::ForwardPlan& plan,
                              const pie_forward::PieForwardOp& op) {
    const auto ps = plan.aux_params(op);
    if (ps.size != 1) {
        throw std::runtime_error(
            "declared arm: an attention dispatch states " +
            std::to_string(ps.size) +
            " scalar arguments, wants 1 (window_left)");
    }
    return static_cast<int>(static_cast<std::int32_t>(ps[0]));
}

// `SplitQkv`: one packed bank into three. Identical in gemma-4 and
// qwen3.5 once both read their operands off the plan; llama_like's is
// this plus a row WINDOW, which is the rectangle's and so stays a
// parameter rather than a second arm.
//
// `win_start` offsets each operand by whole rows -- the peel's tail
// splits the hook-visible rows at their absolute offsets, so the
// full-N consumers see one contiguous buffer. Zero is the plain form.
inline void arm_split_qkv(const ArmCtx& c,
                          const pie_forward::PieForwardOp& op) {
    const auto& plan = c.plan;
    auto& values = c.values;
    const int rows = c.rows;
    const int win_start = c.win_start;
    const auto stream = c.stream;
    const auto ins = plan.inputs(op);
    const auto outs = plan.outputs(op);
    need(ins, 1, "split_qkv inputs");
    need(outs, 3, "split_qkv outputs");
    const int q_w = row_width(plan, outs[0]);
    const int kv_w = row_width(plan, outs[1]);
    const auto row = [&](void* base, int width) -> void* {
        return static_cast<std::uint16_t*>(base) +
               static_cast<std::size_t>(win_start) *
                   static_cast<std::size_t>(width);
    };
    kernels::attn::split_qkv_bf16(
        row(c.row(ins[0]), row_width(plan, ins[0])),
        row(c.row(outs[0]), q_w),
        row(c.row(outs[1]), kv_w),
        row(c.row(outs[2]), kv_w),
        rows, q_w, kv_w, stream);
}

// `Embed`: the token table into the residual stream. All four executors
// hold this identically once their operands come off the plan — only
// the WEIGHT lookup differs, because each family binds its tensors
// through its own store, so the resolved pointer is a parameter.
//
// `token_ids` stays a driver input: the fire's tokens are not a traced
// value.
inline void arm_embed(const ArmCtx& c,
                      const pie_forward::PieForwardOp& op,
                      const std::int32_t* token_ids,
                      const void* table,
                      int vocab) {
    const auto& plan = c.plan;
    const auto outs = plan.outputs(op);
    need(outs, 1, "embed outputs");
    kernels::layout::embed_bf16(token_ids, table, c.row(outs[0]),
                                c.rows, row_width(plan, outs[0]), vocab,
                                c.stream);
}

// `residual_add`: `x += residual`, landing on operand 0 — the `kernel!`
// row aliases the result over it, so the destination is the OUTPUT's
// slot and the addend is operand 1. Both spellings that reach here
// (llama_like's post-norm landing, gpt-oss's MoE landing) are this.
inline void arm_residual_add(const ArmCtx& c,
                             const pie_forward::PieForwardOp& op) {
    const auto& plan = c.plan;
    auto& values = c.values;
    const auto ins = plan.inputs(op);
    const auto outs = plan.outputs(op);
    need(ins, 2, "residual add inputs");
    need(outs, 1, "residual add outputs");
    kernels::norm::residual_add_bf16(
        c.row(outs[0]), c.row(ins[1]),
        static_cast<std::size_t>(c.rows) *
            static_cast<std::size_t>(row_width(plan, outs[0])),
        c.stream);
}

// `Rmsnorm`: the row norm, with the WEIGHT FOLD chosen by the variant
// the statement carries. Gemma folds `(1 + w)` instead of `w` — different
// arithmetic, so a different kernel, but the same signature and the same
// row space, and the variant rides on the wire (`op.param0`).
//
// That is what makes this arm family-blind rather than nearly so: the
// fork is a fact of the STATEMENT, not of the executor, and three of the
// four had hard-coded their deployment's answer to it.
//
// `eps` stays a parameter. It is a config number the trace does not
// carry, which is the same reason the weight pointer is one.
//
// So does `gemma_fold`, and that is the migration showing: a SEMANTIC
// `Rmsnorm` makes its caller read the variant off `op.param0`, while a
// text that states `norm::rmsnorm_gemma_bf16` makes its caller pass
// what the registry already matched. Same arm, and only one of the two
// callers is choosing.
inline void arm_rmsnorm(const ArmCtx& c,
                        const pie_forward::PieForwardOp& op,
                        const void* weight,
                        float eps,
                        bool gemma_fold) {
    const auto& plan = c.plan;
    auto& values = c.values;
    const int rows = c.rows;
    const auto stream = c.stream;
    const auto ins = plan.inputs(op);
    const auto outs = plan.outputs(op);
    need(ins, 1, "rmsnorm inputs");
    need(outs, 1, "rmsnorm outputs");
    const int width = row_width(plan, ins[0]);
    if (gemma_fold) {
        kernels::norm::rmsnorm_gemma_bf16(c.row(ins[0]), weight,
                                          c.row(outs[0]), rows, width,
                                          eps, stream);
    } else {
        kernels::norm::rmsnorm_bf16(c.row(ins[0]), weight,
                                    c.row(outs[0]), rows, width, eps,
                                    stream);
    }
}

// ── the WEIGHT REPRESENTATION axis ─────────────────────────────────
//
// Which storage a projection's weight is in used to be a question the
// DRIVER answered: `make_weight_view(&wb.require(name), layer.q_proj_quant)`
// looked into a per-layer descriptor the statement never mentioned, and
// `gemm::act_x_w` routed on what it found. Eighteen call sites across
// two executors, and every one of them was the driver knowing something
// the declaration did not.
//
// Now the declaration STATES the symbol (`MatW::gemm_symbol`) and NAMES
// the scale tensors (`MatW::scale_names`), so the executor's whole job
// is to bind: the enum below is the registry's match, not a decision.
enum class ScaledRepr { PerTensor, PerChannel, PerGroup, Mxfp4Marlin };

// `y = x @ Wᵀ` over a weight stored some way other than dense bf16.
//
// The statement's weights are `[W, scales, (zeros)]` in that order —
// `MatW::scale_names` derives the last two off the first, which is how
// the loader already finds them, so a caller resolves three names and
// passes three pointers.
//
// The group size is DERIVED from the scale tensor rather than read off a
// descriptor, and that is not the same kind of fact as a kernel choice:
// the symbol already fixed the layout, and `K / (scales per row)` is
// arithmetic on two shapes the plan and the checkpoint both state. If
// they disagree the checkpoint is malformed, which is why it throws
// rather than picking something.
inline void arm_scaled_matmul(const ArmCtx& c,
                              const pie_forward::PieForwardOp& op,
                              ScaledRepr repr,
                              cublasHandle_t handle,
                              const DeviceTensor& w,
                              const DeviceTensor& scales,
                              const DeviceTensor* zeros,
                              float beta) {
    const auto& plan = c.plan;
    auto& values = c.values;
    const auto ins = plan.inputs(op);
    const auto outs = plan.outputs(op);
    need(ins, 1, "scaled matmul inputs");
    need(outs, 1, "scaled matmul outputs");
    const int M = c.rows;
    const int N = row_width(plan, outs[0]);
    const int K = row_width(plan, ins[0]);
    const void* const act = c.row(ins[0]);
    void* const y = c.row(outs[0]);
    const void* const zp = zeros != nullptr ? zeros->data() : nullptr;
    switch (repr) {
    case ScaledRepr::PerTensor:
        kernels::gemm::act_x_wt_tensor_scaled(
            handle, act, w.data(), w.dtype(), w.nbytes(),
            scales.data(), scales.dtype(), scales.numel(), zp,
            y, M, N, K, beta);
        break;
    case ScaledRepr::PerChannel:
        // Row-major `[N, K]`, so a channel is an OUTPUT row: axis 0.
        // The other axis would need `scale_numel == K`, and no
        // checkpoint this driver reads stores it that way.
        kernels::gemm::act_x_wt_channel_scaled(
            handle, act, w.data(), w.dtype(), w.nbytes(),
            scales.data(), scales.dtype(), scales.numel(), zp, 0,
            y, M, N, K, beta);
        break;
    case ScaledRepr::PerGroup: {
        const std::size_t per_row =
            scales.numel() / static_cast<std::size_t>(N > 0 ? N : 1);
        if (per_row == 0 || static_cast<std::size_t>(K) % per_row != 0) {
            throw std::runtime_error(
                "declared arm: a grouped-scaled weight states K=" +
                std::to_string(K) + " over " + std::to_string(N) +
                " rows, which does not divide its " +
                std::to_string(scales.numel()) + " scales");
        }
        kernels::gemm::act_x_wt_grouped_scaled(
            handle, act, w.data(), w.dtype(), w.nbytes(),
            scales.data(), scales.dtype(), scales.numel(), zp,
            static_cast<int>(static_cast<std::size_t>(K) / per_row),
            y, M, N, K, beta);
        break;
    }
    case ScaledRepr::Mxfp4Marlin:
        kernels::gemm::act_x_wt_mxfp4_marlin(
            handle, act, w.data(), w.nbytes(),
            scales.data(), scales.numel(), y, M, N, K, beta);
        break;
    }
}

// ── MLA: the absorbed attention three families share ────────────────
//
// deepseek_v4, glm5 and kimi_k2 all state these, and all three reach
// them today through a hand-written pass. Porting them here rather than
// into one family's executor is the point: what differs between the
// three is the HEAD GEOMETRY, which is config either way, and the
// operands, which the statements give.
//
// The geometry stays explicit parameters for the reason `eps` does --
// `qk_nope_dim`, `v_head_dim` and `kv_lora_rank` are how the absorb
// carves a row, and a row width divided by a head count is that carving
// only once you already know one of them.

// `mla_absorb_q_to_latent`: the query into the latent basis.
//
// One operand, one weight, one result whose shape is
// `[Tokens, heads, kv_lora_rank]` -- so the head count comes off the
// result and only the two INNER dims are parameters.
inline void arm_mla_absorb_q_to_latent(const ArmCtx& c,
                                       const pie_forward::PieForwardOp& op,
                                       cublasHandle_t handle,
                                       const void* kv_b_proj,
                                       int qk_nope_dim,
                                       int v_head_dim) {
    const auto& plan = c.plan;
    const auto ins = plan.inputs(op);
    const auto outs = plan.outputs(op);
    need(ins, 1, "mla absorb-q inputs");
    need(outs, 1, "mla absorb-q outputs");
    const auto& rv = plan.value(outs[0]);
    if (rv.rank != 3) {
        throw std::runtime_error(
            "declared arm: an MLA absorb states rank " +
            std::to_string(rv.rank) + ", wants [Tokens, heads, dim]");
    }
    kernels::gemm::mla_absorb_q_to_latent_bf16(
        handle, c.row(ins[0]), kv_b_proj, c.row(outs[0]),
        c.rows, static_cast<int>(rv.dims[1].value), qk_nope_dim, v_head_dim,
        static_cast<int>(rv.dims[2].value));
}

// `mla_absorb_latent_to_v`: the latent attention output back to V.
//
// The inverse, and the mirror image of the shape above: here the
// result's trailing dim is `v_head_dim` and `kv_lora_rank` is the
// parameter, because that is which end of the absorb each one is.
inline void arm_mla_absorb_latent_to_v(const ArmCtx& c,
                                       const pie_forward::PieForwardOp& op,
                                       cublasHandle_t handle,
                                       const void* kv_b_proj,
                                       int qk_nope_dim,
                                       int kv_lora_rank) {
    const auto& plan = c.plan;
    const auto ins = plan.inputs(op);
    const auto outs = plan.outputs(op);
    need(ins, 1, "mla absorb-v inputs");
    need(outs, 1, "mla absorb-v outputs");
    const auto& rv = plan.value(outs[0]);
    if (rv.rank != 3) {
        throw std::runtime_error(
            "declared arm: an MLA absorb states rank " +
            std::to_string(rv.rank) + ", wants [Tokens, heads, dim]");
    }
    kernels::gemm::mla_absorb_latent_to_v_bf16(
        handle, c.row(ins[0]), kv_b_proj, c.row(outs[0]),
        c.rows, static_cast<int>(rv.dims[1].value), qk_nope_dim,
        static_cast<int>(rv.dims[2].value), kv_lora_rank);
}

// `write_mla_to_pages`: commit the compressed KV row and its rope half.
//
// Two operands, no result -- the cache is the effect, and a cache is
// not a traced value. The CSRs and the layer view stay driver inputs
// for the same reason they do in every KV write arm.
inline void arm_write_mla_to_pages(const ArmCtx& c,
                                   const pie_forward::PieForwardOp& op,
                                   MlaCacheLayerView layer,
                                   const std::uint32_t* qo_indptr,
                                   const std::uint32_t* kv_page_indices,
                                   const std::uint32_t* kv_page_indptr,
                                   const std::uint32_t* kv_last_page_lens,
                                   int num_requests,
                                   const std::uint8_t* row_valid) {
    const auto ins = c.plan.inputs(op);
    need(ins, 2, "mla write inputs");
    kernels::attn::write_mla_to_pages(
        layer, c.row(ins[0]), c.row(ins[1]),
        qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
        c.rows, num_requests, c.stream, row_valid);
}

// `attention_mla`: attention over the latent cache.
//
// Two operands (the latent query and its rope half), one result, and an
// LSE the caller owns -- a second OUTPUT where the statement declares
// one, the caller's scratch otherwise, exactly as gpt-oss's sink
// dispatch reads it.
inline void arm_attention_mla(const ArmCtx& c,
                              const pie_forward::PieForwardOp& op,
                              const kernels::attn::MlaPlanCache& mla_plan,
                              MlaCacheLayerView layer,
                              const std::uint32_t* kv_page_indices,
                              AttentionWorkspaceView attn_ws,
                              float* lse_fallback) {
    const auto& plan = c.plan;
    const auto ins = plan.inputs(op);
    const auto outs = plan.outputs(op);
    need(ins, 2, "mla attention inputs");
    need(outs, 1, "mla attention outputs");
    float* lse = outs.size >= 2
                     ? static_cast<float*>(c.row(outs[1]))
                     : lse_fallback;
    kernels::attn::dispatch_attention_mla_bf16(
        mla_plan, c.row(ins[0]), c.row(ins[1]), layer,
        c.row(outs[0]), kv_page_indices, attn_ws, c.stream, lse);
}

// ── The routed decode GEMVs, and the router in front of them ────────
//
// deepseek_v4, glm5 and kimi_k2 state all three. The pair of GEMVs
// treats each (token, expert) route as a one-row GEMM and indexes the
// expert BANK by the router's own indices, which is why the route
// indices are an OPERAND here and not a buffer the arm goes looking for.

// `topk_sigmoid`: each token's top-k experts and their weights.
//
// One operand (the router logits), two results — the indices and the
// weights, in that order, which is what `record_many` states. `top_k`
// and the expert count come off those results' own shapes; what stays
// a parameter is the deployment's routing policy, because no shape
// carries a renormalize flag or a scaling factor.
inline void arm_topk_sigmoid(const ArmCtx& c,
                             const pie_forward::PieForwardOp& op,
                             int num_experts,
                             const float* correction_bias,
                             bool renormalize,
                             float routed_scaling_factor) {
    const auto& plan = c.plan;
    const auto ins = plan.inputs(op);
    const auto outs = plan.outputs(op);
    need(ins, 1, "topk router inputs");
    need(outs, 2, "topk router outputs");
    kernels::moe::topk_sigmoid_bf16(
        c.row(ins[0]),
        static_cast<std::int32_t*>(c.row(outs[0])),
        static_cast<float*>(c.row(outs[1])),
        correction_bias, c.rows, num_experts,
        row_width(plan, outs[0]), renormalize, routed_scaling_factor,
        c.stream);
}

// `moe_gate_up_gemv` / `moe_down_gemv`: the routed pair.
//
// Operands are `[route_indices, activation]` -- the DSL's order, and
// the kernels' first two arguments in the same order, which is not a
// coincidence: the wrapper was written from the launcher.
//
// The result is `[Tokens, top_k, width]`, so `top_k` and the output
// width both come off it. The REDUCTION dim does not, and stays a
// parameter: it is the operand's row width, which the arm reads.
inline void arm_moe_routed_gemv(const ArmCtx& c,
                                const pie_forward::PieForwardOp& op,
                                bool gate_up,
                                const void* bank) {
    const auto& plan = c.plan;
    const auto ins = plan.inputs(op);
    const auto outs = plan.outputs(op);
    need(ins, 2, "routed gemv inputs");
    need(outs, 1, "routed gemv outputs");
    const auto& rv = plan.value(outs[0]);
    if (rv.rank != 3) {
        throw std::runtime_error(
            "declared arm: a routed GEMV states rank " +
            std::to_string(rv.rank) + ", wants [Tokens, top_k, width]");
    }
    const int top_k = static_cast<int>(rv.dims[1].value);
    const int out_w = static_cast<int>(rv.dims[2].value);
    const int in_w = row_width(plan, ins[1]);
    // The gate_up leg reads the residual-width activation and writes
    // `2 * I_moe`; the down leg reads `I_moe` and writes the residual
    // width. Both launchers take `(H, I_moe)` in that order, so which
    // of the two extents plays which role is the SYMBOL's, not a
    // measurement.
    if (gate_up) {
        kernels::moe::moe_gate_up_decode_gemv_bf16(
            static_cast<const std::int32_t*>(c.row(ins[0])),
            c.row(ins[1]), bank, c.row(outs[0]),
            c.rows, top_k, in_w, out_w / 2, c.stream);
    } else {
        kernels::moe::moe_down_decode_gemv_bf16(
            static_cast<const std::int32_t*>(c.row(ins[0])),
            c.row(ins[1]), bank, c.row(outs[0]),
            c.rows, top_k, out_w, in_w, c.stream);
    }
}

// `mla_prepare`: the two projections into the four operands MLA
// attends over, in ONE launch.
//
// deepseek_v4 and glm5 both state it. Two operands in
// (`kv_a`, `q_b`), four results out (`kv_c`, `k_pe`, `q_nope`, `q_pe`)
// -- the DSL's order, which is the launcher's.
//
// The head count and the nope width come off `q_nope`'s own shape
// (`[Tokens, heads, nope]`); everything else the launcher takes is the
// deployment's -- the norm weight, `eps`, the rope theta and its
// interleave, the source row stride, and the YaRN block where one
// applies. Those stay parameters, like every config number an arm
// takes.
inline void arm_mla_prepare(const ArmCtx& c,
                            const pie_forward::PieForwardOp& op,
                            MlaCacheLayerView layer,
                            const void* kv_a_norm_weight,
                            const std::int32_t* positions,
                            const std::uint32_t* qo_indptr,
                            const std::uint32_t* kv_page_indices,
                            const std::uint32_t* kv_page_indptr,
                            const std::uint32_t* kv_last_page_lens,
                            int num_requests,
                            float eps,
                            float theta,
                            bool interleaved,
                            int kv_a_row_stride,
                            const kernels::attn::YarnOriginalParams* yarn,
                            const std::uint8_t* row_valid) {
    const auto& plan = c.plan;
    const auto ins = plan.inputs(op);
    const auto outs = plan.outputs(op);
    need(ins, 2, "mla prepare inputs");
    need(outs, 4, "mla prepare outputs");
    const auto& qn = plan.value(outs[2]);
    if (qn.rank != 3) {
        throw std::runtime_error(
            "declared arm: mla_prepare's q_nope states rank " +
            std::to_string(qn.rank) + ", wants [Tokens, heads, nope]");
    }
    kernels::attn::mla_prepare_bf16(
        layer, c.row(ins[0]), kv_a_norm_weight, c.row(ins[1]),
        c.row(outs[0]), c.row(outs[1]),
        c.row(outs[2]), c.row(outs[3]),
        positions, qo_indptr, kv_page_indices, kv_page_indptr,
        kv_last_page_lens, c.rows, num_requests,
        static_cast<int>(qn.dims[1].value),
        static_cast<int>(qn.dims[2].value),
        eps, theta, interleaved, kv_a_row_stride, yarn, c.stream, row_valid);
}

// ── the FlashInfer decode dispatch ─────────────────────────────────
//
// Every family states this symbol and every family reaches it
// differently: llama_like has one decode plan, gemma-4 picks between a
// full and a sliding one by layer kind (and BUILDS one when the
// deployment sized none), qwen3.5 keys its own. So the PLAN stays the
// caller's and only the call is here.
//
// That split is the honest one. An arm is shared when its body stops
// naming a family's things; a plan cache is a family's thing, and
// pretending otherwise is what put a wrong theta and a wrong cache slot
// into two earlier merges. The caller resolves, the arm binds.
//
// The LSE is a second OUTPUT where the statement declares one and the
// caller's scratch otherwise -- gpt-oss's sink dispatch is the only
// caller that asks, and it asks by stating two results.
inline void arm_attention_decode(const ArmCtx& c,
                                 const pie_forward::PieForwardOp& op,
                                 const kernels::attn::DecodePlanCache& plan_cache,
                                 KvCacheLayerView kv_view,
                                 const std::uint32_t* page_indices,
                                 const std::uint32_t* page_indptr,
                                 const std::uint32_t* last_page_lens,
                                 AttentionWorkspaceView attn_ws,
                                 int window_left,
                                 float sm_scale,
                                 float* lse_fallback,
                                 void* dst_override = nullptr) {
    const auto& plan = c.plan;
    const auto ins = plan.inputs(op);
    const auto outs = plan.outputs(op);
    need(ins, 1, "decode attention inputs");
    // A guard-region form declares no value: the guard owns it, and the
    // caller hands the destination in.
    void* const dst = dst_override != nullptr
                          ? dst_override
                          : (outs.size > 0 ? c.row(outs[0]) : nullptr);
    if (dst == nullptr) {
        throw std::runtime_error(
            "declared arm: a decode dispatch names no destination");
    }
    float* const lse = outs.size >= 2
                           ? static_cast<float*>(c.row(outs[1]))
                           : lse_fallback;
    kernels::attn::dispatch_attention_flashinfer_decode(
        plan_cache, c.row(ins[0]), kv_view, dst, page_indices,
        page_indptr, last_page_lens, attn_ws, c.stream, window_left,
        /*logits_soft_cap=*/0.f, sm_scale, lse);
}

// The EPILOGUE's compaction, which is the half of `LmHead` every
// executor spells the same way.
//
// A fire whose sampled rows are a strict subset gathers them before the
// projection; anything else multiplies every row. The gather's
// destination belongs to the LOWERING, not to a workspace and not to a
// traced value — see `ValueArena::epilogue_gather` — and the caller
// passes it because only the caller knows whether its executor built
// `flat` before the arms or after.
//
// Returns the activation the projection should read and writes the row
// count through `rows`. The GEMM itself stays with the caller: three of
// the four resolve their head weight differently enough (a name, a
// bound tensor, a quantized view) that passing the result back is
// clearer than passing the resolver in.
inline const void* arm_epilogue_gather(const ArmCtx& c,
                                       const pie_forward::PieForwardOp& op,
                                       void* gathered,
                                       const std::int32_t* logit_row_indices,
                                       int num_logit_rows,
                                       int* rows) {
    const auto& plan = c.plan;
    auto& values = c.values;
    const auto stream = c.stream;
    const auto ins = plan.inputs(op);
    need(ins, 1, "lm_head inputs");
    const void* input = c.row(ins[0]);
    if (logit_row_indices == nullptr || num_logit_rows <= 0 ||
        num_logit_rows >= *rows) {
        return input;
    }
    if (gathered == nullptr) {
        throw std::runtime_error(
            "declared arm: the epilogue compacts rows but the lowering "
            "reserved no scratch for it");
    }
    kernels::layout::gather_bf16_rows(
        static_cast<const std::uint16_t*>(input), logit_row_indices,
        static_cast<std::uint16_t*>(gathered), num_logit_rows,
        row_width(plan, ins[0]), stream);
    *rows = num_logit_rows;
    return gathered;
}

}  // namespace pie_cuda_driver::model::declared
