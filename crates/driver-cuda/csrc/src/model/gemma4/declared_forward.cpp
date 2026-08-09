#include "attention_workspace.hpp"
#include "model/gemma4/declared_forward.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "layout/gather_rows.hpp"
#include "norm/residual_add.hpp"
#include "norm/rmsnorm.hpp"
#include "rope/rope.hpp"
#include "norm/scalar_mul.hpp"
#include "attn/softcap.hpp"
#include "attn/qkv_fused.hpp"
#include "attn/split_packed.hpp"
#include "mlp/swiglu.hpp"
#include "layout/embed.hpp"
#include "attn/kv_paged.hpp"
#include "attn/attention_flashinfer.hpp"
#include "attn/attention_naive_paged.hpp"
#include "gemm/gemm.hpp"
#include "model/declared/arms.hpp"
#include "model/declared/execute.hpp"
#include "model/declared/registry.hpp"
#include "model/declared/weights.hpp"
#include "model/declared/value_arena.hpp"
#include <string>
#include <string_view>

namespace pie_cuda_driver::model {

namespace {

// The launcher registry — every kernel a gemma-4 class trace may STATE,
// one enum value per symbol. Deliberately EXHAUSTIVE against the traced
// decode plan: `gemma4_validate_stated_kernels` walks the plan at load
// and a symbol outside this list is a model-load failure, so this list
// and `family::gemma4_cuda` are two spellings of one vocabulary.


}  // namespace

void gemma4_validate_stated_kernels(const pie_forward::ForwardPlan& plan) {
    const std::size_t n = plan.op_count();
    for (std::size_t i = 0; i < n; ++i) {
        const auto& op = plan.op(i);
        if (op.kind != pie_forward::PieForwardOpKind::Launch) continue;
        (void)declared::resolve_kernel(plan.weight_name(op));
    }
}

}  // namespace pie_cuda_driver::model

namespace pie_cuda_driver::model {

namespace {

using pie_forward::PieForwardOp;
using pie_forward::PieForwardOpKind;

// THE HOST ASSIGNS. gemma-4 takes its activation buffers from
// `Buffers::assign` wherever it placed a value; the pin table below now
// speaks only for what the host DECLINED to place.
//
// `PIE_DECLARED_HOST_ARENA=0` puts the pin table back in charge. Kept
// because the next three families convert against it, and because it is
// the A/B that turned a wrong answer into two named facts.
//
// WHAT IT COST TO GET HERE, because the shape of the search is the
// transferable part. The flip produced DETERMINISTIC garbage -- not
// uninitialised memory in the random sense, but some value read from a
// different address than it was written to. Deterministic is
// bisectable, and the axis you bisect on is the whole problem:
//
//   by VALUE ID     unsound. Values that share bytes must move
//                   together, and an id range splits them. It accused
//                   value 0, which is only the residual stream's first
//                   link.
//   by OFFSET       sound but blunt. Sharing bytes is sharing an
//                   offset, so a window cannot split a chain -- but one
//                   offset hosts many chains over a fire (slot 20992
//                   held eleven values), so it bottomed out at a slot
//                   and could not name a statement.
//   by OWNER        both. Every member of a chain has one owner, so the
//                   window is chain-safe; two chains reusing a slot
//                   have different owners, so it separates them.
//
// On the owner axis it took nine runs to name each of two bugs, and
// both were the same KIND of thing: an op that writes over its operand
// while the table describing it said it produced a value. Semantic rope
// (`kernels::semantic_in_place`) and the logit softcap (`in_place` on
// its `kernel!` row). Under the pin table both aliases were the same
// workspace field, so neither had ever been observable.
//
// Two earlier candidates were ruled out by fixing them and re-running,
// and both fixes stand on their own: the PLE relay, whose geglu signal
// read `per_layer_token` while the transpose filling it wrote to the
// arena; and seam pinning, which inferred the exposed set from a
// neighbouring op's INPUTS and so never pinned a value exposed as an
// OUTPUT.
bool gemma4_host_arena_enabled() {
    const char* v = std::getenv("PIE_DECLARED_HOST_ARENA");
    return v == nullptr || v[0] != '0';
}

// `PIE_DECLARED_HOST_ARENA_LO` / `_HI`: let the host place only the
// values whose OWNER falls in `[lo, hi)`, and pin the rest as before.
//
// The failure is DETERMINISTIC, which makes it bisectable, and this is
// the cut. Not by value id, which was the first attempt and was
// unsound: values that share bytes -- an in-place chain, a select
// window -- must move together, and an id range splits them. It
// reported value 0 as the culprit, which is just the residual stream's
// first link; placing it while its own later links stayed pinned put
// the stream in two buffers.
//
// The second attempt cut by OFFSET, which cannot split a chain --
// sharing bytes IS sharing an offset -- and that got the first bug.
// But it could not NAME the second, because the converse fails: one
// offset hosts many chains over a fire (slot 20992 held eleven values),
// so a window that isolates an offset still admits everything that ever
// reused it, and the bisect bottomed out at a slot rather than a
// statement.
//
// So cut by OWNER, which is the axis that was wanted both times. Every
// member of a chain has the same owner, so a window is still
// chain-safe; and two chains that reuse a slot have DIFFERENT owners,
// so the window separates what the offset axis conflated. `value_owner`
// is what `Buffers::assign` already computes to get liveness right --
// this only asks the host to say it out loud.
// `PIE_DECLARED_ARENA_ZERO=1`: clear the activation block before the
// fire. A DISCRIMINATOR, not a fix.
//
// A per-role workspace buffer holds the previous fire's contents in any
// row a kernel skips; a packed arena holds another value's. If some
// statement reads bytes its producer never wrote, the convention
// supplies something shape-compatible and the arena supplies noise --
// so zeroing changes the output. If instead a value is simply being
// read from the wrong ADDRESS, zeroing changes nothing about which
// address that is, and the garbage stays put.
//
// One run tells the two apart, which is worth more than another
// hypothesis.
bool arena_zero_enabled() {
    const char* v = std::getenv("PIE_DECLARED_ARENA_ZERO");
    return v != nullptr && v[0] == '1';
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

[[noreturn]] void throw_drift(const std::string& what) {
    throw std::runtime_error("declared gemma4: " + what +
                             " (the trace and the driver drifted)");
}

// A plan weight name split into layer and field — llama_like's parse.
struct ParsedName {
    int layer = -1;
    std::string_view field;
};

ParsedName parse_name(std::string_view name) {
    constexpr std::string_view prefix = "layer.";
    if (name.substr(0, prefix.size()) != prefix) return {-1, name};
    const std::size_t dot = name.find('.', prefix.size());
    if (dot == std::string_view::npos) throw_drift("weight name '" +
                                                   std::string(name) + "'");
    int layer = 0;
    for (std::size_t i = prefix.size(); i < dot; ++i) {
        if (name[i] < '0' || name[i] > '9') {
            throw_drift("weight name '" + std::string(name) + "'");
        }
        layer = layer * 10 + (name[i] - '0');
    }
    return {layer, name.substr(dot + 1)};
}

// The binder. gemma-4's trace names its weights after the driver's own
// fields, so this is a map and not a translation.
// The signature is `declared::WeightBinder`'s -- a plain function
// pointer plus context -- so no arm that goes through it names a struct
// field, which is what lets an arm be shared.
const DeviceTensor* bind_gemma4_weight(
    const void* ctx, const declared::ParsedWeightName& nm,
    std::string_view name) {
    const auto& w = *static_cast<const Gemma4Weights*>(ctx);
    if (nm.layer < 0) {
        if (nm.field == "embed") return w.embed;
        if (nm.field == "embed_per_layer") return w.embed_per_layer;
        if (nm.field == "ple_model_proj") return w.ple_model_proj;
        if (nm.field == "ple_model_norm") return w.ple_model_norm;
        if (nm.field == "final_norm") return w.final_norm;
        if (nm.field == "lm_head") return w.lm_head;
        throw_drift("unknown model weight '" + std::string(name) + "'");
    }
    if (nm.layer >= static_cast<int>(w.layers.size())) {
        throw_drift("weight names layer " + std::to_string(nm.layer));
    }
    const Gemma4LayerWeights& l = w.layers[static_cast<std::size_t>(nm.layer)];
    if (nm.field == "attn_norm") return l.attn_norm_pre;
    if (nm.field == "post_attn_norm") return l.attn_norm_post;
    if (nm.field == "pre_ffw_norm") return l.mlp_norm_pre;
    if (nm.field == "post_ffw_norm") return l.mlp_norm_post;
    if (nm.field == "qkv") return l.qkv_proj_fused;
    if (nm.field == "q_proj") return l.q_proj;
    if (nm.field == "k_proj") return l.k_proj;
    if (nm.field == "v_proj") return l.v_proj;
    if (nm.field == "o_proj") return l.o_proj;
    if (nm.field == "q_norm") return l.q_norm;
    if (nm.field == "k_norm") return l.k_norm;
    if (nm.field == "gate_up") return l.gate_up_proj_fused;
    // The unfused pair, for a deployment without the packed bank (E2B).
    if (nm.field == "gate_proj") return l.gate_proj;
    if (nm.field == "up_proj") return l.up_proj;
    if (nm.field == "down") return l.down_proj;
    if (nm.field == "ple_gate") return l.ple_input_gate;
    if (nm.field == "ple_proj") return l.ple_projection;
    if (nm.field == "ple_norm") return l.ple_norm;
    throw_drift("unknown layer weight '" + std::string(name) + "'");
}


}  // namespace

std::string gemma4_validate_stated_weights(
    const pie_forward::ForwardPlan& plan, const Gemma4Weights& w) {
    // The name-resolution DRY WALK, qwen3_5's precedent. Without it an
    // unbound weight is discovered by the first fire and takes the whole
    // MODEL LOAD down; with it the plan simply declines and the
    // hand-written pass runs. That difference is what makes arming this
    // drive by default safe on a geometry nobody has booted yet — E2B
    // needed exactly this treatment three times.
    const auto resolves = [&](std::string_view name) {
        if (name.empty()) return true;
        // Names the executor does NOT resolve to a tensor: `scale.*` is a
        // CONSTANT riding in the weight slot so the arm can tell four
        // identical launches apart.
        if (name.rfind("scale.", 0) == 0) return true;
        try {
            return declared::WeightBinder{&bind_gemma4_weight, &w}
                       .find(name) != nullptr;
        } catch (const std::exception&) {
            return false;
        }
    };
    for (std::size_t i = 0; i < plan.op_count(); ++i) {
        const auto& op = plan.op(i);
        // On a LAUNCH op the weight slot holds the KERNEL SYMBOL, not a
        // weight — the arms read `aux_names` for the weights. Checking
        // the symbol here declined every deployment for a bogus reason,
        // which a fault-injection run caught: the drive was silently off
        // and the parity gate still said PASS, because both sides were
        // then the hand-written pass.
        const std::string_view name =
            op.kind == pie_forward::PieForwardOpKind::Launch
                ? std::string_view{}
                : plan.weight_name(op);
        if (!resolves(name)) {
            return "weight '" + std::string(name) + "' unresolvable";
        }
        if (op.kind == pie_forward::PieForwardOpKind::Launch) {
            const auto aux = plan.aux_names(op);
            for (std::size_t j = 0; j < aux.size; ++j) {
                const std::string_view a = plan.name(aux[j]);
                if (!resolves(a)) {
                    return "weight '" + std::string(a) + "' unresolvable";
                }
            }
        }
    }
    return {};
}

bool gemma4_forward_declared(
    const Gemma4DeclaredPlan& declared,
    const Gemma4Weights& w,
    const HfConfig& cfg,
    const Gemma4ForwardCfg& fwd_cfg,
    Workspace& ws,
    Gemma4MoeMlpWorkspace& moe_ws,
    KvCache& cache,
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
    const std::uint8_t* row_valid_d,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows)
{
    if (!declared.usable) return false;
    // Every weight an arm reads goes through the binder: the arms name
    // what the TRACE names, never a struct field.
    const declared::WeightBinder wb{&bind_gemma4_weight, &w};
    // WHICH CLASS. `use_decode_path` is the hand-written pass's own test
    // and this mirrors it, `force_prefill_path` included — a deployment
    // forced onto the prefill kernels must reach the PREFILL class here
    // or the drive would fire a decode dispatch the hand pass never
    // would.
    const bool decode_class = is_pure_decode && !fwd_cfg.force_prefill_path;
    const pie_forward::ForwardPlan& plan =
        decode_class ? declared.decode : declared.prefill;
    if (!decode_class && (qo_indptr_h == nullptr || kv_page_indptr_h == nullptr)) {
        // The prefill class's two dispatches both read host indptrs.
        return false;
    }
    // Say ONCE per class that this drive took a fire of it. Without this
    // line "the output is coherent" is evidence about the hand-written
    // pass as easily as about this one — an eligibility gate that
    // silently answers false looks exactly like a drive that works.
    {
        static std::atomic<bool> said[2] = {{false}, {false}};
        const std::size_t slot = decode_class ? 0 : 1;
        if (!said[slot].exchange(true)) {
            std::fprintf(stderr,
                         "[declared-gemma4] first %s fire: N=%d R=%d ops=%zu\n",
                         decode_class ? "DECODE" : "PREFILL",
                         total_tokens, num_requests, plan.op_count());
        }
    }

    const int N = total_tokens;
    const int R = num_requests;
    const int H = cfg.hidden_size;
    // The NARROW MLP width. A double-wide deployment (E2B) doubles it
    // on the KV-shared layers, so every MLP arm reads the layer's own
    // width — `w.per_layer_intermediate` is what the binder measured off
    // the gate_proj tensor, and it is the same number the trace baked in.
    const int I = cfg.intermediate_size;
    const int V = cfg.vocab_size;
    const int L = cfg.num_hidden_layers;
    const int ple_dim = cfg.gemma_hidden_size_per_layer_input;
    const float eps = cfg.rms_norm_eps;
    cudaStream_t stream = cublas.stream();

    // The PLE relay's two buffers. Without them the prologue has nowhere
    // to land, and the hand-written pass allocates on the fly — a shape
    // this drive does not reproduce, so it declines instead.
    if (moe_ws.ple_token.empty() || moe_ws.ple_proj.empty()) return false;
    if (!cache.format().is_native_bf16()) return false;

    // The value the previous statement produced, by slot. gemma-4's
    // buffers are the hand-written pass's, so the drive threads them the
    // way that pass does rather than allocating an arena: `ws.y` is the
    // residual stream, `ws.norm_x` the block scratch.
    void* per_layer_token = moe_ws.ple_token.data();
    void* per_layer_proj = moe_ws.ple_proj.data();
    // A decode plan built for THIS fire, where the deployment cached
    // none for the layer kind. Declared out here so it outlives the
    // dispatch that reads it; see `decode_plan_for` in the walk.
    kernels::attn::DecodePlanCachePtr lazy_decode_plan;
    int lm_head_rows = N;

    // Layer state the arms need, refreshed as the drive walks into a
    // layer. The op's own `layer` field carries it, so nothing here
    // counts.
    int cur_layer = -1;
    bool cur_full = false;
    bool cur_shared = false;
    int cur_d = 0;
    int cur_hq = 0;
    int cur_hk = 0;
    int cur_inter = 0;
    const auto enter = [&](int l) {
        if (l < 0 || l == cur_layer) return;
        cur_layer = l;
        cur_full = w.layers[static_cast<std::size_t>(l)].is_full;
        cur_shared = w.layers[static_cast<std::size_t>(l)].is_shared;
        cur_d = w.per_layer_head_dim[static_cast<std::size_t>(l)];
        cur_hq = cfg.num_attention_heads * cur_d;
        cur_hk = w.per_layer_num_kv_heads[static_cast<std::size_t>(l)] * cur_d;
        cur_inter =
            (static_cast<std::size_t>(l) < w.per_layer_intermediate.size())
                ? w.per_layer_intermediate[static_cast<std::size_t>(l)]
                : I;
    };

    // The FULL layers' partial-rotary width, the driver's derivation.
    const auto rotary_of = [&](int l) {
        const float f =
            w.per_layer_partial_rotary_factor[static_cast<std::size_t>(l)];
        const int d = w.per_layer_head_dim[static_cast<std::size_t>(l)];
        return std::max(2, 2 * static_cast<int>(0.5f * f * d));
    };

    const auto gemm = [&](const void* act, std::string_view weight, void* out,
                          int m, int n, int k, float beta) {
        kernels::gemm::act_x_wt_bf16(cublas.handle(), act, wb.require(weight).data(),
                                out, m, n, k, beta);
    };

    // Declared HERE so the arms can capture it; filled after the
    // lowering exists, which is the only thing that has to come first.
    declared::ValueArena values;

    // A traced value's ROW WIDTH: the product of every dim but the
    // leading one, which is the row axis. This is the number the arms
    // used to carry as `cur_hq`, `cur_hk`, `cur_inter`, `L * ple_dim`
    // and `H` — per-layer bookkeeping the executor maintained beside a
    // declaration that already said it.
    //
    // Returns 0 when a dim after the first is not a constant, which
    // happens: a rank-3 value whose middle axis is `Tokens`
    // (`[N, L, ple_dim]`) has no fixed row width at all. No arm below
    // asks for one of those, and the ones that would are gated in
    // `model/tests/arena_soundness.rs` by name.
    // An operand span is a VIEW into the plan's flat id array, so
    // indexing past its end reads the next statement's operands and
    // hands an arm a plausible pointer to the wrong buffer. Every arm
    // that takes a fixed arity states it here instead.
    const auto need = [&](const auto& span, std::size_t n, const char* what) {
        if (span.size < n) {
            throw std::runtime_error(
                std::string("declared gemma4: ") + what + " states " +
                std::to_string(span.size) + " operands, needs " +
                std::to_string(n));
        }
    };

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

    // The epilogue's gather destination, filled once the lowering
    // exists. Hoisted because this executor builds `flat` after the
    // arms.
    void* epi_gather = nullptr;

    const auto execute_op = [&](const PieForwardOp& op) {
        enter(op.layer);
        switch (op.kind) {
        case PieForwardOpKind::Embed: {
            // ISLAND (value arena). Two sites differing only in WHERE
            // the rows land and how wide they are -- both the trace's.
            // `token_ids` stays a driver input: it is the fire's, not a
            // traced value.
            const std::string_view name = plan.weight_name(op);
            const auto outs = plan.outputs(op);
            need(outs, 1, "embed outputs");
            declared::arm_embed({plan, values, N, 0, stream}, op,
                                token_ids, wb.require(name).data(), V);
            break;
        }
        case PieForwardOpKind::Matmul: {
            // ISLAND (value arena). Twelve branches told apart by the
            // WEIGHT NAME used to sit here, and every one of them chose
            // buffers and widths the trace already states: the operands
            // are `op.inputs[0]` and `op.outputs[0]`, and a GEMM's two
            // extents are those values' row widths. Reading the name to
            // rediscover them was the family convention doing work the
            // declaration had already done.
            //
            // The widths come off the value descriptors rather than the
            // `cur_*` per-layer bookkeeping, which is the same number by
            // a shorter route — a traced value's trailing dims ARE its
            // row width, and for these statements that is `cur_hq`,
            // `2 * cur_inter`, `L * ple_dim` and the rest, per layer,
            // without the executor tracking any of it.
            //
            // The `throw_drift` on an unrecognised field goes with them,
            // and is not a guard lost: it fired when the DECLARATION
            // named a matmul this arm had no placement for, and there is
            // no placement left to lack. A weight that does not exist
            // still refuses, one line down, where `gemm` requires it.
            const std::string_view name = plan.weight_name(op);
            const auto ins = plan.inputs(op);
            const auto outs = plan.outputs(op);
            need(ins, 1, "matmul inputs");
            need(outs, 1, "matmul outputs");
            gemm(values.slot(ins[0]), name, values.slot(outs[0]),
                 N, row_width(outs[0]), row_width(ins[0]), 0.f);
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

        case PieForwardOpKind::RmsnormPerHead: {
            // ISLAND (value arena), and three branches become one.
            //
            // They differed in nothing but their extents, and the
            // extents were re-derived per site from config: `N * L` by
            // `ple_dim`, `N * heads` by `cur_d`, `N * (hk / d)` by
            // `cur_d`. All three are the same statement — split the
            // value's row into HEAD-WIDE rows — and the head width is
            // `param0`, which the op has carried the whole time.
            const std::string_view name = plan.weight_name(op);
            const auto ins = plan.inputs(op);
            const auto outs = plan.outputs(op);
            need(ins, 1, "per-head norm inputs");
            need(outs, 1, "per-head norm outputs");
            const int head = static_cast<int>(op.param0);
            if (head <= 0) {
                throw_drift("per-head norm on '" + std::string(name) +
                            "' states no head width");
            }
            kernels::norm::rmsnorm_bf16(
                values.slot(ins[0]), wb.require(name).data(),
                values.slot(outs[0]), N * (row_width(ins[0]) / head), head,
                eps, stream);
            break;
        }
        case PieForwardOpKind::SplitQkv: {
            // SHARED ARM (D1). Identical here and in qwen3.5 once both
            // read their operands off the plan, so it exists once.
            declared::arm_split_qkv({plan, values, N, 0, stream}, op);
            break;
        }
        case PieForwardOpKind::Rope: {
            // Only the FULL layers reach the semantic kind: sliding
            // layers state the fused rounded pair instead.
            //
            // The head COUNTS stay read from config: rope needs the
            // rotation's head geometry, and a value's row width divided
            // by the head dim is that geometry only once you know the
            // head dim, which this op does not state. The BUFFERS are
            // the trace's, which is the half that was convention.
            const auto outs = plan.outputs(op);
            need(outs, 2, "rope outputs");
            kernels::rope::rope_partial_bf16(
                values.slot(outs[0]), values.slot(outs[1]), positions, N,
                cfg.num_attention_heads, cur_hk / cur_d, cur_d,
                static_cast<int>(op.param1),
                w.per_layer_rope_theta[static_cast<std::size_t>(cur_layer)],
                stream);
            break;
        }
        case PieForwardOpKind::LmHead: {
            const std::string_view name = plan.weight_name(op);
            const auto ins = plan.inputs(op);
            const auto outs = plan.outputs(op);
            need(ins, 1, "lm_head inputs");
            need(outs, 1, "lm_head outputs");
            // SHARED ARM (D1): the compaction is identical in three
            // executors; only the head weight's resolution is not.
            int rows = N;
            const void* const input = declared::arm_epilogue_gather(
                {plan, values, N, 0, stream}, op, epi_gather,
                logit_row_indices_d, num_logit_rows, &rows);
            lm_head_rows = rows;
            gemm(input, name, values.slot(outs[0]), rows, V, H, 0.f);
            break;
        }
        case PieForwardOpKind::Launch: {
            const std::string_view sym = plan.weight_name(op);
            const auto names = plan.aux_names(op);
            const auto aux = [&](std::size_t i) {
                return plan.name(names[i]);
            };
            // WHICH DECODE PLAN THIS LAYER'S DISPATCH TAKES, resolved
            // before the shared switch rather than inside an arm.
            //
            // Two cached plans, and which one is the layer's: a
            // full-attention layer and a sliding one are planned
            // differently, and no statement says which kind a layer is.
            // Where the deployment cached neither, one is built for this
            // fire and kept alive by `lazy_decode_plan`, which outlives
            // the walk.
            //
            // Asked ONLY of a decode dispatch. Every other op would pay
            // for a plan nothing reads.
            const auto decode_plan_for =
                [&]() -> const kernels::attn::DecodePlanCache* {
                if (declared::resolve_kernel(sym) !=
                    declared::Kernel::AttnFlashinferDecode) {
                    return nullptr;
                }
                if (auto* p = (cur_full ? moe_ws.decode_plan_full
                                        : moe_ws.decode_plan_sliding)
                                  .get()) {
                    return p;
                }
                lazy_decode_plan = kernels::attn::make_decode_plan();
                kernels::attn::plan_attention_flashinfer_decode(
                    *lazy_decode_plan, kv_page_indptr_h, R,
                    cfg.num_attention_heads, cur_hk / cur_d, cur_d,
                    cache.page_size(), attn_ws.view(), stream,
                    /*enable_cuda_graph=*/true,
                    /*full_attention_variant=*/cur_full,
                    cache.hnd_layout());
                return lazy_decode_plan.get();
            };
            // THE SHARED SWITCH FIRST (D1). Every symbol whose arm is
            // family-blind lives in `declared/execute.hpp`; what remains
            // below is this family's RESIDUE. A `false` is an answer --
            // "stated, and this family executes it its own way".
            const declared::ExecCtx ectx{
                {plan, values, N, 0, stream},
                wb, cache, attn_ws, cublas, nullptr,
                /*state_cache=*/nullptr,
                positions, qo_indptr, kv_page_indices, kv_page_indptr,
                kv_last_page_lens, row_valid_d,
                qo_indptr_h, kv_page_indptr_h,
                nullptr, nullptr, R,
                nullptr, false,
                eps, /*sm_scale=*/1.0f, /*lse_fallback=*/nullptr,
                0.f,
                cfg.num_attention_heads, cfg.num_key_value_heads, cur_d, cur_d,
                cur_layer,
                // Resolved per op, which is exactly why the plan is a
                // context field rather than something an arm could reach
                // for. This family states no planned prefill, and every
                // dispatch it states declares its own result.
                decode_plan_for(), /*prefill_plan=*/nullptr,
                /*region_dst=*/nullptr,
            };
            if (declared::execute_shared(ectx, op)) break;
            switch (declared::resolve_kernel(sym)) {
            // The SCALE generates. Four sites named four buffers and
            // four element counts to apply one scalar; the buffer and
            // the count are the value's, and the scalar is the
            // statement's now -- a number in the param channel, where it
            // was a NAME this arm turned back into arithmetic the host
            // had already done.
            case declared::Kernel::LoraQkvCorrection:
                // Unreachable by construction: the `HasLora` guard's
                // then-region needs a lora row, and gemma-4 states none.
                // Loud rather than silent -- an adapter dropped without a
                // word is the failure this whole arc exists to prevent.
                throw std::runtime_error(
                    "declared gemma4: lora correction reached, but gemma-4 "
                    "has no adapter support on either side (arc 2 should "
                    "have declined this fire)");
            // The RELAY TRANSPOSE generates. Its three extents were
            // read from config on the reading that `Tokens` being off
            // the result's leading axis left nothing to derive from --
            // and the leading axis is exactly what carries two of them:
            // the result is `[L, Tokens, ple_dim]`.
            case declared::Kernel::QkvPackedPost: {
                auto kv_view = cache.layer_view(cur_layer);
                kernels::attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16(
                    values.slot(plan.inputs(op)[0]),
                    values.slot(plan.outputs(op)[0]),
                    kv_view.k_pages, kv_view.v_pages,
                    wb.require(aux(0)).data(), wb.require(aux(1)).data(),
                    positions, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, row_valid_d, N,
                    cfg.num_attention_heads, cur_hk / cur_d, cur_d,
                    cache.page_size(), kv_view.hnd_layout,
                    w.per_layer_rope_theta[static_cast<std::size_t>(cur_layer)],
                    eps, stream);
                break;
            }
            case declared::Kernel::QkRmsnormRopeRounded: {
                const bool q_only = names.size == 1;
                const auto outs_r = plan.outputs(op);
                need(outs_r, 1, "rounded qk-norm-rope outputs");
                kernels::rope::qk_rmsnorm_rope_bf16_rounded(
                    values.slot(outs_r[0]),
                    // Q-ONLY states one value, so there is no k to ask
                    // for; the kernel is told `num_kv_heads = 0` below
                    // and never reads it.
                    values.slot(outs_r[outs_r.size > 1 ? 1 : 0]),
                    wb.require(aux(0)).data(),
                    q_only ? nullptr : wb.require(aux(1)).data(),
                    positions, N, cfg.num_attention_heads,
                    q_only ? 0 : cur_hk / cur_d, cur_d,
                    w.per_layer_rope_theta[static_cast<std::size_t>(cur_layer)],
                    eps, stream);
                break;
            }
            case declared::Kernel::RopePartial:
                // NOT the shared arm: this family's theta and rotary
                // width are PER LAYER, and the shared one reads a single
                // context value. Restored deliberately after the merge
                // reached for it -- a wrong theta rotates silently.
                kernels::rope::rope_partial_bf16(
                    values.slot(plan.outputs(op)[0]),
                    values.slot(plan.outputs(op)[0]), positions, N,
                    cfg.num_attention_heads, /*num_kv_heads=*/0, cur_d,
                    rotary_of(cur_layer),
                    w.per_layer_rope_theta[static_cast<std::size_t>(cur_layer)],
                    stream);
                break;
            case declared::Kernel::RmsnormNoScale:
                {
                    const auto outs = plan.outputs(op);
                    need(outs, 1, "v-norm outputs");
                    kernels::norm::rmsnorm_no_scale_bf16(
                        values.slot(outs[0]), values.slot(outs[0]),
                        N * (row_width(outs[0]) / cur_d), cur_d, eps, stream);
                }
                break;
            // The decode dispatch is SHARED. What kept it here was which
            // of two cached plans the layer takes, and that question is
            // answered above `execute_shared` now (`decode_plan_for`).
            // The pre-scaled query rides `sm_scale = 1.0f` in the
            // context, where it already was.
            // The chunked geglu GENERATES: its row states one operand,
            // one result, the fire's rows and the result's width, which
            // is the whole call.
            // The GEGLU PAIR generates, both of its sites. They were
            // told apart by comparing the result's WIDTH against
            // `ple_dim`, because the PLE gate's second operand was the
            // WHOLE relay and the layer offset was this arm's to add.
            // The text states a `select` of the relay now, so the offset
            // is a placement the host makes and the two sites differ
            // only in which values they name.
            case declared::Kernel::NormResidualScaleNorm: {
                const std::string_view first = aux(0);
                const ParsedName nm = parse_name(first);
                // Two sites: the attention landing (norm_x -> y, then the
                // MLP's input norm) and the PLE landing (norm_y -> y, then
                // the NEXT layer's input norm).
                const bool ple = nm.field == "ple_norm";
                // The PLE landing carries the layer's own scalar; the
                // attention landing carries 1. The declaration does not
                // state it — it is a per-layer load-time constant the
                // executor reads the way the hand-written pass does.
                const float scale =
                    ple ? w.layers[static_cast<std::size_t>(cur_layer)]
                              .layer_scalar_value
                        : 1.f;
                const auto ins = plan.inputs(op);
                const auto outs = plan.outputs(op);
                need(ins, 1, "norm-residual-scale-norm inputs");
                need(outs, 2, "norm-residual-scale-norm outputs");
                // `(landed, mlp_in)`: the stream, then the normed
                // activation. The `ple ? norm_y : norm_x` input choice
                // goes with them -- it was this family naming which
                // scratch the previous statement had landed in.
                kernels::norm::rmsnorm_residual_add_scale_rmsnorm_bf16(
                    values.slot(ins[0]), wb.require(first).data(),
                    values.slot(outs[0]), scale,
                    wb.require(aux(1)).data(), values.slot(outs[1]),
                    N, H, eps, stream);
                break;
            }
            // The norm-and-residual-add GENERATES. The `ple_norm` test
            // that used to live here chose between two input scratches,
            // and the trace naming its input is what removed it; the
            // width being the result's is what removed the rest.
            case declared::Kernel::LogitSoftcap:
                kernels::attn::logit_softcap_bf16(
                    values.slot(plan.outputs(op)[0]),
                    fwd_cfg.final_logit_softcap,
                    static_cast<std::size_t>(lm_head_rows) * V, stream);
                break;
            case declared::Kernel::AttnNaivePaged: {
                auto kv_view = cache.layer_view(cur_layer);
                // `num_pages_in_batch` is the host indptr's LAST entry —
                // the fire's page count, not the layer's and not the
                // cache's.
                kernels::attn::attention_naive_paged(
                    values.slot(plan.inputs(op)[0]), kv_view,
                    values.slot(plan.outputs(op)[0]),
                    qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, N, R,
                    static_cast<int>(kv_page_indptr_h[R]),
                    cfg.num_attention_heads, stream,
                    declared::stated_window_left(plan, op),
                    /*sm_scale=*/1.0f, /*logits_soft_cap=*/0.f,
                    /*lse_out=*/nullptr);
                break;
            }
            }
            break;
        }
        case PieForwardOpKind::HookSite:
            // gemma-4's sites are OBSERVATION-only, and arc 2 admits no
            // hooked fire at all (`in.stage_hooks == nullptr` is an
            // eligibility term above), so there is never a program to
            // invoke here. The op is still STATED, because the seam is
            // real and its position is checked -- the arm is what makes
            // the trace executable rather than a load failure.
            //
            // A fire that arrived here WITH hooks would be an eligibility
            // bug, not a site to serve; when gemma-4 grows hook support
            // this is the arm that gains the invoke, next to qwen3_5's.
            break;
        default:
            throw_drift("op kind " +
                        std::to_string(static_cast<std::uint32_t>(op.kind)) +
                        " has no emission rule");
        }
    };

    // Build the fire's rows, lower them, execute the list.
    std::vector<pie_forward::PieForwardRow> rows(static_cast<std::size_t>(N));
    for (int r = 0; r < N; ++r) {
        auto& row = rows[static_cast<std::size_t>(r)];
        row.multi_token = decode_class ? 0 : 1;
        row.custom_mask = 0;
        row.hooked = 0;
        row.lora = 0;
        row.write_desc = 0;
        row.wants_scores = 0;
        row.samples = 1;
        row._pad = 0;
        row.depth_k = -1;
    }
    if (logit_row_indices_d != nullptr && num_logit_rows > 0 &&
        num_logit_rows < N) {
        for (int r = num_logit_rows; r < N; ++r) {
            rows[static_cast<std::size_t>(r)].samples = 0;
        }
    }
    const pie_forward::PieForwardLowered flat =
        plan.lower(rows.data(), rows.size());
    if (flat.uncovered != pie_forward::PieForwardUncovered::None) return false;

    // THE PIN PASS (`model/declared/value_arena.hpp`): this family's
    // buffer convention, stated ONCE.
    //
    // Every arm below used to carry a piece of it — "the normed
    // activation is `ws.norm_x`", "the geglu lands in `ws.gate`" — which
    // is why an arm could not be shared with a family spelling the same
    // role differently (qwen3_5 norms into `ws.norm_x` where llama_like
    // uses `ws.norm_y`). Collected here, an arm asks the arena by VALUE
    // ID and never learns whose convention it is serving.
    //
    // The bytes do not move. Each entry names the buffer that op's arm
    // writes today, so a converted arm addresses exactly what it
    // addressed before and the family A/B is a real comparison rather
    // than a re-baselining. Host-assigned offsets take over per island,
    // as each island's pins come out.
    values.reset_pins_only(plan.value_count());
    values.bind_offsets(ws.declared_values.data(),
                        ws.declared_values.nbytes(), flat);
    epi_gather = values.epilogue_gather(flat);
    declared::trace_arena("gemma4", plan, flat,
                          ws.declared_values.nbytes(), N, R);
    if (arena_zero_enabled()) {
        CUDA_CHECK(cudaMemsetAsync(ws.declared_values.data(), 0,
                                   ws.declared_values.nbytes(), stream));
    }
    const bool host_arena = gemma4_host_arena_enabled();
    const std::size_t arena_lo = host_arena_lo();
    const std::size_t arena_hi = host_arena_hi();
    {
        const std::size_t op_count = plan.op_count();
        for (std::size_t i = 0; i < op_count; ++i) {
            const PieForwardOp& op = plan.op(i);
            const auto outs = plan.outputs(op);
            if (outs.size == 0) continue;
            // THE TABLE WINS where it speaks. This is the line that
            // turns "the host assigns" on, and it is on.
            //
            // While the arms were converting, a pin outranked the host's
            // offset so an unmoved arm's convention still held. gemma-4
            // has no unmoved arms left, so these entries are needed only
            // for what the host DECLINED to place — the values a seam
            // exposes, which machinery outside the walk reaches by name.
            // Everything else now lives where `Buffers::assign` put it.
            //
            // It waited on a NUMBER, not on the arms. The block a fire
            // needed was 6.13 GB for this family, against ~1.3 GB for
            // every workspace field it would replace, so turning it on
            // meant refusing at load. Best-fit-split-coalesce took that
            // to 0.91 GB — under what the fields cost — and
            // `declared_arena_bytes` sizes the block from the
            // declaration instead of from `{N, H + I}`.
            //
            // The table stays rather than shrinking to the seam list: it
            // is the record of what this family's convention WAS, and
            // the next family converts by reading it rather than
            // rederiving it from a hand-written pass.
            const auto pin = [&](std::size_t which, void* ptr) {
                if (which >= outs.size) return;
                const std::uint32_t v = outs[which];
                if (host_arena && v < flat.value_offsets_len) {
                    const std::size_t at = flat.value_offsets[v];
                    // The window is on the OWNER, not on `v` and not on
                    // `at`; a value with no owner table is its own.
                    const std::size_t owner =
                        v < flat.value_owners_len
                            ? static_cast<std::size_t>(flat.value_owners[v])
                            : static_cast<std::size_t>(v);
                    if (at != declared::ValueArena::kNamed &&
                        owner >= arena_lo && owner < arena_hi) {
                        return;
                    }
                }
                values.pin(v, ptr);
            };
            switch (op.kind) {
            case PieForwardOpKind::Embed:
                pin(0, plan.weight_name(op) == "embed" ? ws.y.data()
                                                       : per_layer_token);
                break;
            case PieForwardOpKind::Matmul: {
                const ParsedName nm = parse_name(plan.weight_name(op));
                if (nm.field == "ple_model_proj")   pin(0, per_layer_proj);
                else if (nm.field == "qkv")         pin(0, ws.qkv_fused.data());
                else if (nm.field == "q_proj")      pin(0, ws.q.data());
                else if (nm.field == "k_proj")      pin(0, ws.k.data());
                else if (nm.field == "v_proj")      pin(0, ws.v.data());
                else if (nm.field == "o_proj")      pin(0, ws.norm_x.data());
                else if (nm.field == "gate_up")     pin(0, ws.gate_up_fused.data());
                else if (nm.field == "gate_proj")   pin(0, ws.gate.data());
                else if (nm.field == "up_proj")     pin(0, ws.up.data());
                else if (nm.field == "down")        pin(0, ws.norm_x.data());
                else if (nm.field == "ple_gate")    pin(0, ws.norm_x.data());
                else if (nm.field == "ple_proj")    pin(0, ws.norm_y.data());
                break;
            }
            case PieForwardOpKind::Rmsnorm:
                // The row norms are `Launch` now; their entry moved to
                // the Launch case with the statement. This stays for a
                // semantic trace, which gemma-4's CUDA text no longer
                // produces.
                break;
                break;
            case PieForwardOpKind::RmsnormPerHead: {
                const ParsedName nm = parse_name(plan.weight_name(op));
                if (nm.field == "ple_model_norm") pin(0, per_layer_proj);
                else if (nm.field == "q_norm")    pin(0, ws.q.data());
                else if (nm.field == "k_norm")    pin(0, ws.k.data());
                break;
            }
            case PieForwardOpKind::SplitQkv:
                pin(0, ws.q.data());
                pin(1, ws.k.data());
                pin(2, ws.v.data());
                break;
            case PieForwardOpKind::Rope:
                pin(0, ws.q.data());
                pin(1, ws.k.data());
                break;
            case PieForwardOpKind::LmHead:
                pin(0, ws.logits.data());
                break;
            case PieForwardOpKind::Launch: {
                const auto names = plan.aux_names(op);
                const auto aux = [&](std::size_t j) { return plan.name(names[j]); };
                switch (declared::resolve_kernel(plan.weight_name(op))) {
                case declared::Kernel::RmsnormRow:
                case declared::Kernel::RmsnormRowGemma:
                    // Both sites (`attn_norm`, `final_norm`) norm the
                    // stream into the same scratch.
                    pin(0, ws.norm_x.data());
                    break;
                case declared::Kernel::ResidualAdd:
                    pin(0, per_layer_proj);
                    break;
                case declared::Kernel::ScalarMul: {
                    const std::string_view which = aux(0);
                    if (which == "scale.sqrt_hidden")        pin(0, ws.y.data());
                    else if (which == "scale.sqrt_ple_dim")  pin(0, per_layer_token);
                    else                                     pin(0, per_layer_proj);
                    break;
                }
                case declared::Kernel::TransposeNldToLnd: pin(0, per_layer_token); break;
                case declared::Kernel::QkvPackedPost:     pin(0, ws.q.data()); break;
                case declared::Kernel::QkRmsnormRopeRounded:
                    pin(0, ws.q.data());
                    pin(1, ws.k.data());
                    break;
                case declared::Kernel::RopePartial:
                case declared::Kernel::RopeFull:         pin(0, ws.q.data()); break;
                case declared::Kernel::RmsnormNoScale:    pin(0, ws.v.data()); break;
                case declared::Kernel::AttnFlashinferDecode:
                case declared::Kernel::AttnFlashinferPrefillPlanless:
                case declared::Kernel::AttnNaivePaged:    pin(0, ws.attn_out.data()); break;
                case declared::Kernel::ChunkedGegluTanh:  pin(0, ws.gate.data()); break;
                case declared::Kernel::GegluTanh: {
                    // TWO sites for one kernel, told apart by the WIDTH
                    // the op declares — the same test the arm makes.
                    const auto& val = plan.value(outs[0]);
                    const std::uint32_t width = val.dims[val.rank - 1].value;
                    pin(0, static_cast<int>(width) == ple_dim ? ws.norm_x.data()
                                                              : ws.gate.data());
                    break;
                }
                case declared::Kernel::NormResidualScaleNorm:
                    // `(landed, mlp_in)` in the declaration: the stream
                    // first, the normed activation second.
                    pin(0, ws.y.data());
                    pin(1, ws.norm_x.data());
                    break;
                case declared::Kernel::NormResidualAdd:   pin(0, ws.y.data()); break;
                case declared::Kernel::LogitSoftcap:      pin(0, ws.logits.data()); break;
                default: break;
                }
                break;
            }
            default:
                break;
            }
        }
    }

    std::size_t next_site = 0;
    std::size_t at = 0;
    while (at < flat.launches_len || next_site < flat.structural_len) {
        const bool site_first =
            at >= flat.launches_len ||
            (next_site < flat.structural_len &&
             flat.structural[next_site].at_op < flat.launches[at].at_op);
        if (site_first) {
            execute_op(plan.op(flat.structural[next_site].at_op));
            ++next_site;
            continue;
        }
        const std::uint32_t at_op = flat.launches[at].at_op;
        while (at < flat.launches_len && flat.launches[at].at_op == at_op) ++at;
        execute_op(plan.op(at_op));
    }
    return true;
}

}  // namespace pie_cuda_driver::model
