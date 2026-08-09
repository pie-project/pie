#include "attention_workspace.hpp"
#include "model/llama_like/declared_forward.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "model/declared/value_arena.hpp"
#include "model/declared/arms.hpp"
#include "model/declared/execute.hpp"
#include "model/declared/registry.hpp"
#include "model/declared/weights.hpp"
#include "batch/supergraph.hpp"
#include "norm/add_bias.hpp"
#include "layout/embed.hpp"
#include "layout/gather_rows.hpp"
#include "attn/head_dim_pad.hpp"
#include "attn/kv_paged.hpp"
#include "norm/residual_add.hpp"
#include "norm/rmsnorm.hpp"
#include "rope/rope.hpp"
#include "attn/qkv_fused.hpp"
#include "attn/split_packed.hpp"
#include "mlp/swiglu.hpp"
#include "model/attn_page_mask.hpp"
#include "model/attn_score.hpp"
#include "model/lora.hpp"
#include "model/stage_hooks.hpp"
#include "attn/attention_flashinfer.hpp"
#include "attn/attention_xqa.hpp"
#include "gemm/gemm.hpp"

namespace pie_cuda_driver::model {

namespace {

using pie_forward::PieForwardNormPlacement;
using pie_forward::PieForwardNormVariant;
using pie_forward::PieForwardOp;
using pie_forward::PieForwardOpKind;
using pie_forward::PieForwardQkNorm;
using pie_forward::PieForwardRopeKind;

// The name grammar is `model/declared/weights.hpp`'s — it was identical in
// every family executor, which is the first thing that says these executors
// wanted to be one.
using declared::ParsedWeightName;
using declared::parse_weight_name;
using declared::throw_unknown_weight;

// This family's half of `declared::WeightBinder`: a traced name against
// Qwen3Weights. Every arm goes through it, so no arm names a struct field —
// which is what lets an arm be shared with a family whose field is spelled
// differently (qwen3_5's `attn_norm_pre` is this family's `attn_norm`).
const DeviceTensor* bind_llama_like_weight(
    const void* ctx, const ParsedWeightName& nm, std::string_view name)
{
    const auto& w = *static_cast<const Qwen3Weights*>(ctx);
    if (nm.layer < 0) {
        if (nm.field == "embed") return w.embed;
        if (nm.field == "final_norm") return w.final_norm;
        if (nm.field == "lm_head") return w.lm_head;
        throw_unknown_weight(name);
    }
    if (nm.layer >= static_cast<int>(w.layers.size())) {
        throw_unknown_weight(name);
    }
    const Qwen3LayerWeights& l = w.layers[static_cast<std::size_t>(nm.layer)];
    // The vocabulary is the TRACE's (`crates/model-compiler/src/dsl.rs`'s `Layer`), which
    // is why this table is the family's whole contribution: `gate_up` is one
    // traced name whether or not the checkpoint bound a fused bank, and the
    // arm asks for the split halves by field when it did not.
    if (nm.field == "attn_norm") return l.attn_norm;
    if (nm.field == "mlp_norm") return l.mlp_norm;
    if (nm.field == "q_norm") return l.q_norm;
    if (nm.field == "k_norm") return l.k_norm;
    if (nm.field == "qkv") return l.qkv_proj_fused;
    if (nm.field == "q_proj") return l.q_proj;
    if (nm.field == "k_proj") return l.k_proj;
    if (nm.field == "v_proj") return l.v_proj;
    if (nm.field == "o_proj") return l.o_proj;
    if (nm.field == "q_bias") return l.q_bias;
    if (nm.field == "k_bias") return l.k_bias;
    if (nm.field == "v_bias") return l.v_bias;
    if (nm.field == "gate_up") return l.gate_up_proj_fused;
    if (nm.field == "gate_proj") return l.gate_proj;
    if (nm.field == "up_proj") return l.up_proj;
    if (nm.field == "down") return l.down_proj;
    throw_unknown_weight(name);
}

// This family's PIN PASS: which traced values live in a buffer the rest
// of the driver reaches BY NAME. Stated once, over the plan, instead of
// once per arm — LoRA captures the normed activation's pointer at fire
// setup, the fused decode launch reads the qkv bank, hook sites observe
// the query, the sampler reads the logits. An arm then just asks the
// arena by value id and never learns whose convention it is serving.
void pin_llama_like_values(const pie_forward::ForwardPlan& plan,
                           Workspace& ws,
                           bool post_norm,
                           declared::ValueArena& values)
{
    const std::size_t ops = plan.op_count();
    for (std::size_t i = 0; i < ops; ++i) {
        const PieForwardOp& op = plan.op(i);
        const auto outs = plan.outputs(op);
        if (outs.size == 0) continue;
        switch (op.kind) {
        case PieForwardOpKind::Embed:
        case PieForwardOpKind::ResidualAdd:
            // The residual stream.
            values.pin(outs[0], ws.y.data());
            break;
        case PieForwardOpKind::Rmsnorm:
            // The row norms are `Launch` now (`cuda::rmsnorm` states the
            // fold), so their entries live in the Launch case below.
            // This one stays for a SEMANTIC trace and does nothing for a
            // CUDA one, which no longer carries the kind.
            break;
        case PieForwardOpKind::Matmul: {
            const ParsedWeightName nm = parse_weight_name(plan.weight_name(op));
            if (nm.field == "qkv") {
                values.pin(outs[0], ws.qkv_fused.data());
            } else if (nm.field == "q_proj") {
                values.pin(outs[0], ws.q.data());
            } else if (nm.field == "k_proj") {
                values.pin(outs[0], ws.k.data());
            } else if (nm.field == "v_proj") {
                values.pin(outs[0], ws.v.data());
            } else if (nm.field == "gate_up") {
                // The PACKED binding's destination -- and now the only
                // reading of this name. An unfused binding states its
                // two halves (2d), each a value of its own, so the pair
                // below serves them.
                values.pin(outs[0], ws.gate_up_fused.data());
            } else if (nm.field == "gate_proj") {
                values.pin(outs[0], ws.gate.data());
            } else if (nm.field == "up_proj") {
                values.pin(outs[0], ws.up.data());
            } else if (nm.field == "o_proj" || nm.field == "down") {
                values.pin(outs[0], post_norm ? ws.norm_x.data()
                                              : ws.y.data());
                if (nm.field == "o_proj") {
                    // Pinned by CONSUMER, not producer: a lowered trace
                    // may state its attention as a stated-kernel Launch
                    // rather than the semantic `Attention` op, and the
                    // value it produces still lives in `ws.attn_out`.
                    // Saying it here covers both spellings.
                    const auto ins = plan.inputs(op);
                    if (ins.size > 0) values.pin(ins[0], ws.attn_out.data());
                }
            }
            break;
        }
        case PieForwardOpKind::Peel:
            // A PEEL produces a value the same way a guard does: its two
            // regions run over complementary row ranges and BOTH write
            // the one result, so the peel is the single producer and its
            // regions record no outputs of their own.
            //
            // llama_like's is the hook split over the QUERY -- the
            // attention dispatches and the hook sites downstream read
            // the peel's value, not the fused rope's -- and nothing
            // bound it, which is what "value 17 ... no pin pass bound
            // it" was. The regions write `ws.q`, so that is where the
            // result lives.
            if (outs.size > 0) values.pin(outs[0], ws.q.data());
            break;
        case PieForwardOpKind::Launch: {
            // The FUSED qk-norm+rope rewrites q and k where they lie,
            // exactly as the semantic rope below does, and its results
            // are what the attention and the KV write read. This is the
            // spelling llama_like's own gate model states -- 84 times
            // per decode text -- so unlike the rope entry below, this
            // one is checked.
            const std::string_view sym = plan.weight_name(op);
            if ((sym == "rope::qk_rmsnorm_rope_bf16" ||
                 sym == "rope::qk_rmsnorm_rope_bf16_devwin") &&
                outs.size >= 2) {
                values.pin(outs[0], ws.q.data());
                values.pin(outs[1], ws.k.data());
            }
            // THE ROW NORMS, moved here with the statement. `attn_norm`'s
            // result is LoRA's `qkv_in`, captured at fire setup and
            // therefore reached by NAME from outside this walk — the one
            // reason a converted island still needs an entry. Missing it
            // would not show on a gate: `lora=0` on every fire the
            // harness runs.
            if (sym == "norm::rmsnorm_bf16" ||
                sym == "norm::rmsnorm_gemma_bf16") {
                const auto aux = plan.aux_names(op);
                if (aux.size == 1 && outs.size >= 1) {
                    const ParsedWeightName nm =
                        parse_weight_name(plan.name(aux[0]));
                    if (nm.field == "attn_norm") {
                        values.pin(outs[0], post_norm ? ws.norm_y.data()
                                                      : ws.norm_x.data());
                    } else if (nm.field == "final_norm") {
                        values.pin(outs[0], ws.norm_y.data());
                    }
                    // `mlp_norm`'s result is read only by the gate/up
                    // matmul, so it stays an arena value.
                }
            }
            break;
        }
        case PieForwardOpKind::Rope:
            // Rope rewrites q and k where they lie; the attention and
            // the KV write downstream still name those buffers, so the
            // rotated results have to land on them. Without this the arm
            // would write host-assigned bytes nothing else reads -- and
            // no gate here would say so, because qwen3-0.6b states the
            // FUSED `qk_rmsnorm_rope` and never reaches the arm.
            if (outs.size >= 2) {
                values.pin(outs[0], ws.q.data());
                values.pin(outs[1], ws.k.data());
            }
            break;
        case PieForwardOpKind::RmsnormPerHead: {
            // The per-head q/k norms rewrite the projections where they
            // lie by convention; rope and the attention downstream still
            // name those buffers, so the RESULT has to land on them.
            const ParsedWeightName nm = parse_weight_name(plan.weight_name(op));
            if (outs.size > 0) {
                values.pin(outs[0], nm.field == "k_norm" ? ws.k.data()
                                                         : ws.q.data());
            }
            break;
        }
        case PieForwardOpKind::SplitQkv:
            if (outs.size >= 3) {
                values.pin(outs[0], ws.q.data());
                values.pin(outs[1], ws.k.data());
                values.pin(outs[2], ws.v.data());
            }
            break;
        case PieForwardOpKind::Attention:
            values.pin(outs[0], ws.attn_out.data());
            break;
        case PieForwardOpKind::LmHead:
            values.pin(outs[0], ws.logits.data());
            break;
        default:
            break;
        }
    }
}

const Qwen3LayerWeights& layer_of(
    const Qwen3Weights& w, const ParsedWeightName& nm, std::string_view name)
{
    if (nm.layer < 0 ||
        nm.layer >= static_cast<int>(w.layers.size())) {
        throw_unknown_weight(name);
    }
    return w.layers[nm.layer];
}

const DeviceTensor* require(const DeviceTensor* t, std::string_view name) {
    if (t == nullptr) {
        throw std::runtime_error(
            "declared forward: weight '" + std::string(name) +
            "' is named by the trace but not bound");
    }
    return t;
}

// The DENSE view, and a refusal where `make_weight_view` used to be.
//
// A semantic `Matmul` means one arithmetic over a weight read directly;
// that is the whole of what the kind says, and it fans to exactly one
// kernel. When a checkpoint's weight is stored some other way, the
// DECLARATION says so — `MatW::repr` picks the symbol and names the
// scale tensors, and the Launch arm binds them.
//
// So a layer that carries a quant descriptor while the statement records
// a plain `Matmul` is FACTS DRIFT: the trace was built against a dense
// deployment and this one is not. `make_weight_view` used to absorb that
// silently by routing on the descriptor, which is exactly the shape this
// arc is removing. It throws instead.
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

// The launcher registry's vocabulary: every kernel a class trace may
// STATE as a `Launch` op (dsl::cuda's raw signatures), one enum value per
// launcher symbol. `resolve_launch_kernel` is the registry lookup; the
// executor's Launch arm switches on the result and BINDS — buffers,
// plans, staging — without choosing. A symbol outside this vocabulary
// means the trace and this executor drifted; `build` validates every
// stated symbol at model load so that drift fails at boot, not mid-fire.


// Boot validation: every Launch symbol a class trace states must resolve.
void validate_stated_kernels(const pie_forward::ForwardPlan& plan) {
    const std::size_t n = plan.op_count();
    for (std::size_t i = 0; i < n; ++i) {
        const PieForwardOp& op = plan.op(i);
        if (op.kind == PieForwardOpKind::Launch) {
            (void)declared::resolve_kernel(plan.weight_name(op));
        }
    }
}

// Row-offset views into `[N, width]` bf16 buffers — the Peel regions'
// window binding (A3), the hand-written `bf16_row` twins. Offset zero is
// the identity, so the windowed call forms serve the unwindowed walk too.
inline void* bf16_row(void* base, int row, int width) {
    return static_cast<std::uint16_t*>(base) +
           static_cast<std::size_t>(row) * width;
}
inline const void* bf16_row(const void* base, int row, int width) {
    return static_cast<const std::uint16_t*>(base) +
           static_cast<std::size_t>(row) * width;
}

// Rung 3 (north-star-dsl.md): the static C++ form of the class traces,
// emitted by `cargo run -p pie-forward --bin emit-cuda` and committed.
// Uses the helpers above (require, dense); the digest constant
// it defines names the deployment it was emitted from, and the dispatch
// in `llama_like_forward_declared` runs it only on exact match.
#include "model/llama_like/generated/qwen3_0_6b.inc"
#include "model/llama_like/generated/olmo2_1b.inc"
#include "model/llama_like/generated/qwen2_5_1_5b.inc"
#include "model/llama_like/generated/mistral_7b_v03.inc"
#include "model/llama_like/generated/phi3_mini.inc"

// PIE_DECLARED_FORWARD_GENERATED=1 routes digest-matched fires through
// the generated static form instead of the interpreter walk — the third
// leg of the parity proof (hand-written ≡ interpreter ≡ generated).
bool generated_forward_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_DECLARED_FORWARD_GENERATED");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

}  // namespace

namespace {

// Why a DEPLOYMENT has no declared plan at all.
//
// The fire-level `DeclineReason` (`llama_like_model.cpp`) says which
// fires fall back; this says which DEPLOYMENTS never had the choice.
// They are different questions with different owners: a decline is
// driver work, and one of these is DSL VOCABULARY — the trace has no way
// to say the thing, so the hand-written body is the only executor there
// and cannot be deleted until it does.
//
// It was a comment with an ellipsis in it ("TP, quantized projections,
// non-standard rope, ..."), which is not a work list. Each `return out`
// in the gate below now carries its name, and the name is printed once
// per model load — loud, unconditional, and cheap, because a deployment
// traces once.
enum class NoPlanReason {
    None,
    RopeNotStandard,      // YaRN / M-RoPE
    TensorParallel,       // shards claimed, no communicator bound
    LayerBinding,         // no layers, or a count that disagrees with the config
    QuantizedProjection,  // QuantMeta WeightViews the trace does not describe
    MixedProjectionRepr,  // two storage kinds where the facts carry one
    MixedFusedQkv,        // a per-layer split that makes the fused_qkv fact a lie
    QkvBiasUnbound,       // the config says bias, the tensors did not arrive
    QkNormConvention,     // a q/k-norm weight shape that names no convention
};

const char* no_plan_name(NoPlanReason r) {
    switch (r) {
    case NoPlanReason::None:                return "none";
    case NoPlanReason::RopeNotStandard:     return "rope-not-standard";
    case NoPlanReason::TensorParallel:      return "tensor-parallel";
    case NoPlanReason::LayerBinding:        return "layer-binding";
    case NoPlanReason::QuantizedProjection: return "quantized-projection";
    case NoPlanReason::MixedProjectionRepr: return "mixed-projection-repr";
    case NoPlanReason::MixedFusedQkv:       return "mixed-fused-qkv";
    case NoPlanReason::QkvBiasUnbound:      return "qkv-bias-unbound";
    case NoPlanReason::QkNormConvention:    return "qk-norm-convention";
    }
    return "?";
}

}  // namespace

bool llama_like_bands_apply(const LlamaLikeDeclaredPlan& declared,
                            const LlamaLikePlanState& plan_state,
                            bool is_pure_decode) {
    if (plan_state.depth_band_count < 2) return false;
    // Bands describe a PURE-DECODE fire's rows — the hand path's rule
    // (`bands_runnable`), and `derive_depth_bands` enforces it from the
    // other side by refusing any region table with a multi-token region.
    if (!is_pure_decode) return false;
    // And the fire's own class has to state the axis, or the arms have
    // nowhere to put a band.
    const auto& plan = is_pure_decode ? declared.decode : declared.prefill;
    return static_cast<bool>(plan) && plan.view().depth_window != 0;
}

LlamaLikeDeclaredPlan build_llama_like_declared_plan(
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    const Qwen3Weights& w,
    const KvCache& cache)
{
    LlamaLikeDeclaredPlan out;
    // An empty plan is not an error and never was — it means "the
    // hand-written body is this deployment's executor". Saying WHICH
    // rule sent it there costs one line at load and turns the gate into
    // a work list.
    const auto refuse = [](NoPlanReason reason) -> LlamaLikeDeclaredPlan {
        std::fprintf(stderr,
                     "[declared] no plan for this deployment: reason=%s "
                     "(the hand-written body serves it)\n",
                     no_plan_name(reason));
        return LlamaLikeDeclaredPlan{};
    };

    // Representability gate: everything the v0 trace has no vocabulary for
    // returns empty and the model keeps the hand-written path. Each line
    // names the hand-written feature it stands in for.
    if (fwd_cfg.rope_kind != RopeKind::Standard) {
        return refuse(NoPlanReason::RopeNotStandard);  // YaRN/M-RoPE
    }
    // Post-norm placement (olmo2/olmo3) is admitted: the trace carries the
    // matmul(beta=0) → rmsnorm → residual_add triplet and the executor
    // launches the hand-written post-norm block's kernels.
    // Qwen-2 bias is admitted (OpKind::AddBias since the qwen2_5 rung):
    // the trace states the three broadcast adds after the lora guard and
    // before norms/rope, the executor launches the hand-written
    // `maybe_add_bias` kernels. Guarded below on the tensors actually
    // being bound.
    // TENSOR PARALLELISM is admitted (3). The trace states its own
    // shard widths and the two landings that recombine them, so what
    // used to be `NoPlanReason::TensorParallel` -- "all-reduces the
    // trace does not state" -- is a thing the trace states.
    //
    // What the deployment still owes is the communicator: a rank that
    // says it shards and bound none has no way to run the collectives,
    // and that is a binding fault rather than an unstated kernel.
    if (fwd_cfg.tp_size > 1 && fwd_cfg.tp_comm == nullptr) {
        return refuse(NoPlanReason::TensorParallel);
    }
    // Padded head_dim (Phi-3-mini, 96 → 128) is admitted: the pad/strip
    // launches around KV-write/attention are emitter knowledge (the trace
    // speaks the logical head_dim throughout), handled in the executor
    // exactly as the hand-written `head_dim_padded` branches.
    if (w.layers.empty() ||
        w.layers.size() != static_cast<std::size_t>(cfg.num_hidden_layers)) {
        return refuse(NoPlanReason::LayerBinding);
    }
    const bool fused_qkv = w.layers[0].qkv_proj_fused != nullptr;
    // The deployment's WEIGHT REPRESENTATION, read once off layer 0's
    // binding. This is the whole of what `make_weight_view` used to ask
    // per call site, asked once and handed to the DECLARATION instead
    // (`LlamaLikeCudaFacts::proj_repr`), which then states a symbol per
    // projection and names the scale tensors it needs.
    //
    // Read off `o_proj` because it is the one projection every
    // configuration binds separately -- a fused-QKV deployment has no
    // `q_proj` and a packed-gate_up one no `gate_proj`, so either would
    // read as dense on a checkpoint that is not.
    const std::optional<QuantMeta>& repr_meta = w.layers[0].o_proj_quant;
    for (const auto& layer : w.layers) {
        // A mixed fused/unfused binding would make the single
        // `fused_qkv` fact a lie -- and so would a mixed
        // representation make `proj_repr` one. The declaration carries
        // ONE answer per deployment, so a deployment with two is
        // refused by name rather than half-stated.
        const auto same_repr = [&](const std::optional<QuantMeta>& m) {
            if (m.has_value() != repr_meta.has_value()) return false;
            if (!m.has_value()) return true;
            return m->kind == repr_meta->kind &&
                   m->group_size == repr_meta->group_size &&
                   m->channel_axis == repr_meta->channel_axis &&
                   (m->zero_point != nullptr) ==
                       (repr_meta->zero_point != nullptr);
        };
        if (!same_repr(layer.q_proj_quant) || !same_repr(layer.k_proj_quant) ||
            !same_repr(layer.v_proj_quant) || !same_repr(layer.o_proj_quant) ||
            !same_repr(layer.gate_proj_quant) ||
            !same_repr(layer.up_proj_quant) ||
            !same_repr(layer.down_proj_quant)) {
            return refuse(NoPlanReason::MixedProjectionRepr);
        }
        if ((layer.qkv_proj_fused != nullptr) != fused_qkv) {
            return refuse(NoPlanReason::MixedFusedQkv);
        }
        // A bias config whose tensors did not bind would make the traced
        // AddBias ops unlaunchable; a bias-less config with stray bias
        // tensors would mean the fact lies the other way.
        if (fwd_cfg.use_qkv_bias &&
            (layer.q_bias == nullptr || layer.k_bias == nullptr ||
             layer.v_bias == nullptr)) {
            return refuse(NoPlanReason::QkvBiasUnbound);
        }
    }
    // q/k-norm convention, from the bound tensor shape — the same evidence
    // the hand-written `rmsnorm_qk` dispatches on, resolved once here
    // because the trace states the convention as a fact:
    //   * per-head (qwen3): weight `[head_dim]` → RmsnormPerHead ops;
    //   * global (olmo2): weight `[heads * head_dim]` → plain row Rmsnorm
    //     over the flattened projection.
    // Anything else (mixed conventions across layers, an unexpected shape)
    // falls back to the hand-written path.
    PieForwardQkNorm qk_norm = PieForwardQkNorm::Off;
    if (fwd_cfg.use_qk_norm) {
        const int Hq_w = cfg.num_attention_heads * cfg.head_dim;
        const int Hk_w = cfg.num_key_value_heads * cfg.head_dim;
        const auto convention_of =
            [&](const DeviceTensor* t, int flat) -> PieForwardQkNorm {
            if (t == nullptr || t->shape().size() != 1) {
                return PieForwardQkNorm::Off;  // no representable convention
            }
            if (t->shape()[0] == cfg.head_dim) return PieForwardQkNorm::PerHead;
            if (t->shape()[0] == flat) return PieForwardQkNorm::Global;
            return PieForwardQkNorm::Off;
        };
        qk_norm = convention_of(w.layers[0].q_norm, Hq_w);
        if (qk_norm == PieForwardQkNorm::Off) {
            return refuse(NoPlanReason::QkNormConvention);
        }
        for (const auto& layer : w.layers) {
            if (convention_of(layer.q_norm, Hq_w) != qk_norm ||
                convention_of(layer.k_norm, Hk_w) != qk_norm) {
                return refuse(NoPlanReason::QkNormConvention);
            }
        }
    }

    pie_forward::PieForwardLlamaLikeFacts facts{};
    facts.hidden = static_cast<std::uint32_t>(cfg.hidden_size);
    facts.layers = static_cast<std::uint32_t>(cfg.num_hidden_layers);
    facts.q_heads = static_cast<std::uint32_t>(cfg.num_attention_heads);
    facts.kv_heads = static_cast<std::uint32_t>(cfg.num_key_value_heads);
    facts.head_dim = static_cast<std::uint32_t>(cfg.head_dim);
    facts.intermediate = static_cast<std::uint32_t>(cfg.intermediate_size);
    facts.vocab = static_cast<std::uint32_t>(cfg.vocab_size);
    facts.rope = static_cast<std::uint32_t>(PieForwardRopeKind::Standard);
    facts.norm_variant =
        static_cast<std::uint32_t>(PieForwardNormVariant::Plain);
    facts.norm_placement = static_cast<std::uint32_t>(
        fwd_cfg.norm_placement == NormPlacement::Post
            ? PieForwardNormPlacement::Post
            : PieForwardNormPlacement::Pre);
    facts.qk_norm = static_cast<std::uint32_t>(qk_norm);
    facts.fused_qkv = fused_qkv ? 1 : 0;
    facts.qkv_bias = fwd_cfg.use_qkv_bias ? 1 : 0;
    // A binding fact, like fused_qkv: bind_llama_like aliases lm_head to
    // embed when the checkpoint ties them, so pointer equality is the truth.
    facts.tied_embeddings = (w.lm_head == w.embed) ? 1 : 0;

    out.plan = pie_forward::ForwardPlan::trace_llama_like(facts);
    out.fused_qkv = fused_qkv;

    // The CUDA backend facts, derived ONCE from this deployment — the
    // same terms the executor's `use_*_path` booleans and the fused
    // predicate computed per fire, now computed here and handed to the
    // declaration, whose class arms STATE the kernels (rung 2,
    // north-star-dsl.md). Term provenance on each line.
    pie_forward::PieForwardLlamaLikeCudaFacts cuda{};
    // context.cpp:1419 derives fwd_cfg.use_xqa_decode (env + kernel
    // support + all-full-attention); prepare adds the cache-format terms
    // (llama_like.cpp:693) — all load-time.
    cuda.xqa_decode = (fwd_cfg.use_xqa_decode &&
                       cache.format().is_native_bf16() &&
                       !cache.hnd_layout())
                          ? 1
                          : 0;
    // The fused decode-QKV epilogue's load-time terms
    // (declared_forward.cpp's old peephole predicate, minus what the
    // build gate above already excluded: bias, per-head-shape checks —
    // qk_norm was resolved from the bound shapes right here).
    cuda.decode_fused_post = (decode_fused_post_enabled() &&
                              cache.format().is_native_bf16() &&
                              cfg.head_dim == cfg.head_dim_kernel &&
                              // The fused epilogue has no bias step —
                              // the hand-written predicate's term, here
                              // since the build gate admits bias now.
                              !fwd_cfg.use_qkv_bias)
                                 ? 1
                                 : 0;
    // workspace.cpp:33 allocates ws.rope_table unconditionally; the
    // executor still checks emptiness loudly at the table launch.
    cuda.rope_table = 1;
    cuda.force_prefill_path = fwd_cfg.force_prefill_path ? 1 : 0;
    // Load-time: the kernel head dim the attention runs at vs the logical
    // one (Phi-3-mini pads 96 -> 128; llama_like.cpp's head_dim_padded).
    cuda.head_dim_padded = (cfg.head_dim != cfg.head_dim_kernel) ? 1 : 0;
    // The WIDTH the pads and the strip state (2c). Zero when the
    // attention runs at the logical head dim, which is what makes
    // `head_dim_padded` exactly `head_dim_kernel != 0`.
    // THE SLIDING WINDOW, per layer, handed to the declaration so the
    // dispatch statements can carry it (`dsl::cuda::attn_at`'s params).
    // What the executors read instead was this same array, at every
    // dispatch, through a config nothing stated.
    //
    // The empty case broadcasts the config's single `sliding_window`:
    // `window_left_at` reads a one-element list for every layer, which
    // is exactly what the drivers' `: fwd_cfg.sliding_window` fallback
    // meant. The vector outlives the trace calls below.
    std::vector<std::int32_t> window_left(
        fwd_cfg.per_layer_window_left.begin(),
        fwd_cfg.per_layer_window_left.end());
    // A NEGATIVE single window is the same statement as no list at
    // all, and the two must not print differently: the digest carries
    // this list, so an empty-vs-`[-1]` disagreement between the two
    // printers would mean no generated TU ever matched.
    if (window_left.empty() && fwd_cfg.sliding_window >= 0) {
        window_left.push_back(
            static_cast<std::int32_t>(fwd_cfg.sliding_window));
    }
    cuda.window_left = window_left.data();
    cuda.window_left_len = static_cast<std::uint32_t>(window_left.size());
    cuda.tp_size = static_cast<std::uint32_t>(
        fwd_cfg.tp_size > 0 ? fwd_cfg.tp_size : 1);
    cuda.head_dim_kernel =
        cuda.head_dim_padded
            ? static_cast<std::uint32_t>(cfg.head_dim_kernel)
            : 0u;
    // The MLP's gate_up BINDING (qwen3.cpp: the loader installs the
    // packed bank when `contract.hpp::dense_fused_projection_joins`
    // accepts the group — it declines quantized and non-BF16 ones). The
    // executor used to re-derive this per layer as
    // `gate_up_proj_fused != nullptr && !ws.gate_up_fused.empty()`; the
    // second term is dead (workspace.cpp allocates that buffer
    // unconditionally, by its own comment), so the binding alone decides
    // and it is known here. Layer 0 speaks for the deployment because
    // the contract accepts or declines a GROUP uniformly — and the
    // executor's per-launch cross-check refuses if a later layer ever
    // disagrees, rather than trusting this line.
    cuda.gate_up_fused =
        (!w.layers.empty() && w.layers[0].gate_up_proj_fused != nullptr) ? 1 : 0;
    // The WEIGHT REPRESENTATION, from the binding read above. This is
    // the line that replaced `NoPlanReason::QuantizedProjection`: the
    // deployment used to be refused here because the trace could not
    // describe its weights, and now it describes them.
    //
    // The payload rides beside the tag rather than in a union, matching
    // the wire (`PieForwardWeightRepr`). `group_size` and `channel_axis`
    // are the checkpoint's own numbers -- the loader read them out of
    // the quantization config -- so nothing here derives anything.
    if (!repr_meta.has_value()) {
        cuda.proj_repr =
            static_cast<std::uint32_t>(pie_forward::PieForwardWeightRepr::Bf16);
    } else {
        switch (repr_meta->kind) {
        case QuantMeta::Kind::PerTensor:
            cuda.proj_repr = static_cast<std::uint32_t>(
                pie_forward::PieForwardWeightRepr::ScaledPerTensor);
            break;
        case QuantMeta::Kind::PerChannel:
            cuda.proj_repr = static_cast<std::uint32_t>(
                pie_forward::PieForwardWeightRepr::ScaledPerChannel);
            break;
        case QuantMeta::Kind::PerGroup:
            cuda.proj_repr = static_cast<std::uint32_t>(
                pie_forward::PieForwardWeightRepr::ScaledPerGroup);
            break;
        }
        cuda.proj_zero_point = repr_meta->zero_point != nullptr ? 1 : 0;
        cuda.proj_group = static_cast<std::uint32_t>(repr_meta->group_size);
        cuda.proj_axis = static_cast<std::uint32_t>(repr_meta->channel_axis);
    }

    out.decode = pie_forward::ForwardPlan::trace_llama_like_cuda(
        facts, cuda, pie_forward::PieForwardFireClass::Decode);
    out.prefill = pie_forward::ForwardPlan::trace_llama_like_cuda(
        facts, cuda, pie_forward::PieForwardFireClass::Prefill);
    // Drift between the declaration's stated kernels and this executor's
    // registry fails at model load, not mid-fire.
    validate_stated_kernels(out.decode);
    validate_stated_kernels(out.prefill);

    // The digest naming what these traces were taken from — the same
    // format `pie_forward::emit_cuda::facts_digest` embeds in the
    // generated .inc (one format, two printers; the three-way parity gate
    // holds them together). A generated TU runs only on exact match.
    out.facts_digest =
        "llama_like/h" + std::to_string(facts.hidden) +
        "/l" + std::to_string(facts.layers) +
        "/qh" + std::to_string(facts.q_heads) +
        "/kvh" + std::to_string(facts.kv_heads) +
        "/hd" + std::to_string(facts.head_dim) +
        "/i" + std::to_string(facts.intermediate) +
        "/v" + std::to_string(facts.vocab) +
        "/rope" + std::to_string(facts.rope) +
        "/nv" + std::to_string(facts.norm_variant) +
        "/np" + std::to_string(facts.norm_placement) +
        "/qk" + std::to_string(facts.qk_norm) +
        "/fq" + std::to_string(facts.fused_qkv) +
        "/te" + std::to_string(facts.tied_embeddings) +
        "/qb" + std::to_string(facts.qkv_bias) +
        "/xqa" + std::to_string(cuda.xqa_decode) +
        "/dfp" + std::to_string(cuda.decode_fused_post) +
        "/rt" + std::to_string(cuda.rope_table) +
        "/fpp" + std::to_string(cuda.force_prefill_path) +
        "/pad" + std::to_string(cuda.head_dim_padded) +
        // The WEIGHT REPRESENTATION -- see the Rust printer for why the
        // payload beside it is deliberately not in here.
        "/pr" + std::to_string(cuda.proj_repr) +
        "/hdk" + std::to_string(cuda.head_dim_kernel) +
        // The shard count: a rank's text states ITS widths, so a trace
        // taken at one tp_size is a different body at another.
        "/tp" + std::to_string(cuda.tp_size) +
        // The SLIDING WINDOW list -- a constant of the emitted text, so
        // a body emitted against one must not serve another.
        "/wl" + [&] {
            std::string out;
            for (std::uint32_t i = 0; i < cuda.window_left_len; ++i) {
                if (i != 0) out += ".";
                out += std::to_string(cuda.window_left[i]);
            }
            return out;
        }();
    return out;
}

// `PIE_DECLARED_FLAT_TRACE=1`: print what each fire's list served. A
// diagnostic, never a switch — there is nothing left to switch between.
//
// It replaces `PIE_DECLARED_SHADOW`, which armed the walk-vs-lowering
// comparison that carried this migration (47% agreement on its first
// run, 100% for the two increments before the drive was written, and
// three real defects found on the way). With the walk gone the shadow
// has nothing to compare against; what survives it is this line and the
// `devwin` count, which the capture-split probe needs to show it took
// the path at all.
bool flat_trace_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("PIE_DECLARED_FLAT_TRACE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

// The rest of the executor's diagnostics, read ONCE.
//
// These were `std::getenv` calls in the fire path — four of them, on
// every fire, for lines that print on none of them. An environment
// lookup is not free and, more to the point, it is not what this file
// does anywhere else: `declared_forward_enabled`, `flat_trace_enabled`
// and `generated_forward_enabled` all latch. A diagnostic that costs
// something when disarmed is a diagnostic people turn off for the wrong
// reason.
bool forward_trace_enabled() {
    static const bool on = std::getenv("PIE_DECLARED_FORWARD_TRACE") != nullptr;
    return on;
}

bool spatial_mask_trace_enabled() {
    static const bool on = std::getenv("PIE_SPATIAL_MASK_TRACE") != nullptr;
    return on;
}

bool lora_fire_trace_enabled() {
    static const bool on = std::getenv("PIE_LORA_FIRE_TRACE") != nullptr;
    return on;
}

// ONE FIRE'S ROWS — the executor's input, and now its only one.
//
// This is where a fire stops being a bundle of driver words and becomes
// what the lowering takes: a row per token, each carrying the axes it
// sits on. Everything downstream is a function of this array and the
// declaration.
//
// Row-level truth where the driver has it and fire-level where it does
// not, which is honest rather than lossy: the axes that are fire-wide
// TODAY (mask, lora, write descriptors) are fire-wide in the trace's
// guards too, so a fire-wide fill selects exactly the arms a per-row one
// would. The genuinely per-row axes are the hook peel (`fast_rows` ends
// the hook-free prefix), the spatial mask split, and depth.
//
// It was born as `shadow_rows`, feeding the comparison that proved the
// lowering agreed with the walk. The walk is gone and the name went
// with it; the function did not have to change at all, which is the
// clearest statement of what that comparison was for.
std::vector<pie_forward::PieForwardRow> fire_rows(
    int n_fire,
    int fast_rows,
    int depth_k,
    int depth_split,
    bool depth_union,
    const std::uint32_t* band_k,
    const std::uint32_t* band_rows,
    std::size_t band_count,
    int mask_split,
    bool has_mask,
    bool has_lora,
    bool has_write_desc,
    bool wants_scores,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows)
{
    std::vector<pie_forward::PieForwardRow> rows(static_cast<std::size_t>(std::max(n_fire, 0)));
    const bool compact =
        logit_row_indices_d != nullptr && num_logit_rows > 0 &&
        num_logit_rows < n_fire;
    for (std::size_t r = 0; r < rows.size(); ++r) {
        pie_forward::PieForwardRow& row = rows[r];
        // The spatial split's UNMASKED PREFIX is a row axis, not a
        // fire flag: `[0, mask_split)` attends causally and the suffix
        // takes the custom dispatch. Filling the mask fire-wide made
        // the lowering see an empty prefix and skip statements the walk
        // ran — the shadow's own first finding, about the shadow.
        row.custom_mask =
            (has_mask && (mask_split < 0 || static_cast<int>(r) >= mask_split))
                ? 1
                : 0;
        row.lora = has_lora ? 1 : 0;
        row.write_desc = has_write_desc ? 1 : 0;
        row.wants_scores = wants_scores ? 1 : 0;
        row.hooked = static_cast<int>(r) >= fast_rows ? 1 : 0;
        // THREE truncation shapes, and each states itself differently.
        //
        // BANDED (>= 2 distinct k): `band_rows[j]` is how many rows are
        // still live at depths >= `band_k[j]`, deepest-first. Inverting
        // that per row: rows below `band_rows[0]` are full depth, and a
        // row at or past `band_rows[j]` dies at `band_k[j]` for the
        // LAST j it reaches. Not filling this at all was the shadow's
        // largest remaining drift — a banded fire read as untruncated,
        // so the lowering covered every layer the bands had retired.
        //
        // UNION: full-depth rows first, truncated after `depth_split`.
        // UNIFORM: every row at `k`, and no split exists to read.
        if (band_count >= 2) {
            int k = -1;
            for (std::size_t j = 0; j < band_count; ++j) {
                if (static_cast<std::uint32_t>(r) >= band_rows[j]) {
                    k = static_cast<int>(band_k[j]);
                }
            }
            row.depth_k = k;
        } else {
            const bool truncated =
                depth_k >= 0 &&
                (!depth_union || static_cast<int>(r) >= depth_split);
            row.depth_k = truncated ? depth_k : -1;
        }
        // Compact fires sample the LAST row of each request; the walk
        // does not carry the index list on the host, so the shadow says
        // "the last `num_logit_rows` rows", which is that set whenever
        // the requests are equal-length and a reported row-range drift
        // otherwise. Stated so a reader does not mistake it for truth.
        row.samples = compact
            ? (static_cast<int>(rows.size() - r) <= num_logit_rows ? 1 : 0)
            : 1;
    }
    return rows;
}


void llama_like_forward_declared(
    const LlamaLikeDeclaredPlan& declared,
    const Qwen3Weights& w,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    const LlamaLikePlanState& plan_state,
    Workspace& ws,
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
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows,
    const std::uint32_t* w_page_d,
    const std::uint32_t* w_off_d,
    const std::uint8_t* row_valid_d,
    bool has_write_desc,
    int runtime_window_left,
    const std::uint8_t* custom_mask_d,
    const std::int32_t* custom_mask_indptr_d,
    const StageHooks* stage_hooks,
    const LoraTable* lora,
    const std::uint32_t* peel_window_d,
    std::uint32_t unmasked_prefix_rows,
    const std::uint32_t* mask_suffix_qo_indptr_d,
    const std::uint32_t* mask_suffix_kv_page_indptr_d,
    std::uint32_t declared_max_layers,
    std::uint32_t declared_full_depth_rows)
{
    // Every weight an arm reads goes through the binder (see its header):
    // the arms name what the TRACE names, never a struct field.
    const declared::WeightBinder wb{&bind_llama_like_weight, &w};
    // Rung 3: the static form, when opted in and emitted for exactly this
    // deployment. Byte-for-byte the same launches as the interpreter walk
    // below — the parity gate's third leg proves it — with every choice
    // resolved at EMISSION time instead of at walk time.
    if (generated_forward_enabled() &&
        declared.facts_digest != kGeneratedDigest_qwen3_0_6b &&
        declared.facts_digest != kGeneratedDigest_olmo2_1b &&
        declared.facts_digest != kGeneratedDigest_qwen2_5_1_5b &&
        declared.facts_digest != kGeneratedDigest_mistral_7b_v03 &&
        declared.facts_digest != kGeneratedDigest_phi3_mini &&
        forward_trace_enabled()) {
        // Silent non-engagement is this path's failure mode; say why.
        std::fprintf(stderr,
                     "[declared-forward-generated] digest mismatch:\n"
                     "  live:    %s\n  emitted: %s | %s\n",
                     declared.facts_digest.c_str(),
                     kGeneratedDigest_qwen3_0_6b,
                     kGeneratedDigest_olmo2_1b);
        std::fprintf(stderr, "  emitted: %s\n",
                     kGeneratedDigest_qwen2_5_1_5b);
        std::fprintf(stderr, "  emitted: %s\n",
                     kGeneratedDigest_mistral_7b_v03);
        std::fprintf(stderr, "  emitted: %s\n",
                     kGeneratedDigest_phi3_mini);
    }
    // A1 (the class-collapse amendment): a custom mask no longer picks a
    // class — the decode/prefill traces carry it as their HasCustomMask
    // guard arm, so the generated static form serves masked fires too
    // (the mask data crosses as arguments).
    // The static form covers EVERY fire the digest matches (rung 3
    // complete): the emitter constructs the lora staging AND the hook
    // sidebands (page mask, score captures) and spells the sites,
    // brackets and corrections with constant layers.
    if (generated_forward_enabled()) {
        const auto run = [&](auto decode_fn, auto prefill_fn) {
            (is_pure_decode ? decode_fn : prefill_fn)(
                w, cfg, fwd_cfg, plan_state, ws, cache, attn_ws, cublas,
                token_ids, positions, qo_indptr,
                kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                total_tokens, num_requests,
                logit_row_indices_d, num_logit_rows,
                w_page_d, w_off_d, row_valid_d, has_write_desc,
                custom_mask_d, custom_mask_indptr_d,
                stage_hooks, lora, peel_window_d,
                unmasked_prefix_rows, mask_suffix_qo_indptr_d,
                declared_max_layers, declared_full_depth_rows);
        };
        if (declared.facts_digest == kGeneratedDigest_qwen3_0_6b) {
            run(generated_llama_like_decode_qwen3_0_6b,
                generated_llama_like_prefill_qwen3_0_6b);
            return;
        }
        if (declared.facts_digest == kGeneratedDigest_olmo2_1b) {
            run(generated_llama_like_decode_olmo2_1b,
                generated_llama_like_prefill_olmo2_1b);
            return;
        }
        if (declared.facts_digest == kGeneratedDigest_qwen2_5_1_5b) {
            run(generated_llama_like_decode_qwen2_5_1_5b,
                generated_llama_like_prefill_qwen2_5_1_5b);
            return;
        }
        if (declared.facts_digest == kGeneratedDigest_mistral_7b_v03) {
            run(generated_llama_like_decode_mistral_7b_v03,
                generated_llama_like_prefill_mistral_7b_v03);
            return;
        }
        if (declared.facts_digest == kGeneratedDigest_phi3_mini) {
            run(generated_llama_like_decode_phi3_mini,
                generated_llama_like_prefill_phi3_mini);
            return;
        }
    }
    // Rung 2 + A2: the fire's SHAPE picks its trace, and the trace
    // states every kernel — attachments (mask, hooks) are its guard
    // arms, nothing below derives a path (north-star-dsl.md).
    const pie_forward::ForwardPlan& plan =
        is_pure_decode ? declared.decode : declared.prefill;
    // The fire's extents. They used to have MUTABLE peers (`N`, `R`)
    // that a depth window rebound per op; a rectangle's row count is
    // that number now, so the only extents left are the fire's own.
    const int N_fire = total_tokens;
    const int R_fire = num_requests;
    const bool depth_stated = plan.view().depth_window != 0;
    const int depth_k =
        depth_stated && declared_max_layers != 0xffffffffu &&
        declared_max_layers <
            static_cast<std::uint32_t>(cfg.num_hidden_layers)
            ? static_cast<int>(declared_max_layers)
            : -1;
    const bool depth_union = depth_k >= 0 &&
        declared_full_depth_rows != 0xffffffffu;
    const int depth_split =
        depth_union ? static_cast<int>(declared_full_depth_rows) : N_fire;
    if (depth_union &&
        (depth_split <= 0 || depth_split >= R_fire ||
         !plan_state.depth_prefix_decode_plan)) {
        throw std::runtime_error(
            "depth union (declared): planned split without a usable "
            "prefix plan (gate drift)");
    }
    // ④ Act 1 (banded depth): >= 2 distinct-k bands stamped by the
    // prepare. Deepest-first arrays; at layer L the live rows are
    // N_fire below the shallowest k, else the matched band's start row
    // (0 = the op is skipped — nothing lives at that depth). Banded
    // fires derive max_layers FULL, so the union/uniform logic above
    // stays idle.
    const int band_count = static_cast<int>(plan_state.depth_band_count);
    // NOT `band_count >= 2 && depth_stated` computed here: the model's
    // eligibility gate has to make the same decision, and when the two
    // were separate copies the gate's went stale one commit after it was
    // written. One function, both callers.
    const bool depth_banded =
        llama_like_bands_apply(declared, plan_state, is_pure_decode);
    if (depth_banded) {
        for (int j = 0; j < band_count; ++j) {
            if (plan_state.depth_band_rows[static_cast<std::size_t>(j)] >
                    0 &&
                !plan_state.depth_band_plans[static_cast<std::size_t>(j)]) {
                throw std::runtime_error(
                    "depth bands (declared): stamped band without a "
                    "usable prefix plan (the model gate should have "
                    "kept this fire hand-written)");
            }
        }
        if (spatial_mask_trace_enabled()) {
            std::fprintf(stderr, "[depth-bands-declared] R=%d m=%d\n",
                         R_fire, band_count);
        }
    }
    // The SSA value arena (`model/declared/value_arena.hpp`): values an arm
    // asks for by id rather than by this family's workspace name.
    //
    // PINS ONLY, because this executor still walks OPS: every value its
    // moved islands ask for is one the pin pass below binds, and the
    // buffers come from the workspace exactly as they did. What changed
    // is who may CHOOSE bytes nobody pinned — the host does now
    // (`Buffers::assign`), and an arena that answered by allocating was
    // a second allocator over the same plan. The islands take
    // host-assigned offsets when this walk moves onto rectangles; until
    // then an unpinned ask names itself as unbound instead of inventing
    // an address.
    declared::ValueArena values;
    values.reset_pins_only(plan.value_count());
    // The Peel split (A3): the hook-free prefix row count — the
    // hand-written `fast_rows` derivation verbatim. A runtime INPUT of
    // the stated Peel op, not a choice: with no hooks every row is the
    // prefix; the dispatch proved rows [0, fast_rows) belong to no
    // attention-stage program.
    const int fast_rows = stage_hooks == nullptr
        ? R_fire
        : std::min(static_cast<int>(stage_hooks->hook_free_prefix_rows),
                   R_fire);
    // Parity-harness visibility (PIE_HOOK_PREFIX_TRACE's pattern): without
    // it a silent fallback to the hand-written path would be
    // indistinguishable from a passing A/B run; fast_rows is what proves
    // a MIXED fire walked the declared Peel.
    if (forward_trace_enabled()) {
        std::fprintf(stderr,
                     "[declared-forward] N=%d R=%d decode=%d fast_rows=%d "
                     "mask=%d hooked=%d lora=%d ops=%zu\n",
                     N_fire, R_fire, is_pure_decode ? 1 : 0, fast_rows,
                     custom_mask_d != nullptr ? 1 : 0,
                     stage_hooks != nullptr ? 1 : 0,
                     (lora != nullptr && lora->usable()) ? 1 : 0,
                     plan.op_count());
    }
    const int H = cfg.hidden_size;
    const int Hq = cfg.num_attention_heads * cfg.head_dim;
    const int Hk = cfg.num_key_value_heads * cfg.head_dim;
    const int I = cfg.intermediate_size;
    const int V = cfg.vocab_size;
    const int d = cfg.head_dim;
    const int dk = cfg.head_dim_kernel;  // padded HEAD_DIM the attention
                                         // kernel runs at (Phi-3: 96 → 128)
    const bool head_dim_padded = (d != dk);
    const int num_q_heads = cfg.num_attention_heads;
    const int num_kv_heads = cfg.num_key_value_heads;
    const float eps = cfg.rms_norm_eps;
    // Post-norm placement (olmo2): decides which buffer each projection
    // reads/writes and how the block norms route — the same buffer walk as
    // the hand-written `post_norm` branches, stated once here.
    const bool post_norm = fwd_cfg.norm_placement == NormPlacement::Post;
    // This family's conventions, stated once over the plan (see the pass).
    pin_llama_like_values(plan, ws, post_norm, values);
    // WHICH VALUE AN OP BINDS: the enclosing value-producing guard's
    // result.
    //
    // The attention's own launches declare no output, and their guard
    // does -- `o_proj` reads v3, written by the BODY guard at op 3,
    // whose 41-op span contains every attention spelling. That is the
    // ABI's phi: "the guard's outputs are the ONE producer whichever
    // region runs; region launches bind the same output buffer and
    // record no outputs of their own."
    //
    // Regions are flat and consecutive (`param0` arms, `[kind, payload,
    // len]` each plus a trailing else length), so the span is
    // computable. Guards NEST here -- op 3's body contains the lora,
    // write-kv and attention chains -- so this walks outermost-first and
    // lets a narrower guard's write overwrite.
    std::vector<std::uint32_t> binds(plan.op_count(),
                                     pie_forward::PIE_FORWARD_NO_VALUE);
    for (std::size_t gi = 0; gi < plan.op_count(); ++gi) {
        const PieForwardOp& g = plan.op(gi);
        if (g.kind != PieForwardOpKind::Guard) continue;
        const auto gouts = plan.outputs(g);
        if (gouts.size == 0) continue;
        const auto run = plan.aux_names(g);
        const std::uint32_t arms = g.param0;
        if (run.size < static_cast<std::size_t>(arms) * 3 + 1) continue;
        std::size_t span = 0;
        for (std::uint32_t a = 0; a < arms; ++a) span += run[a * 3 + 2];
        span += run[arms * 3];
        for (std::size_t j = gi + 1; j <= gi + span && j < binds.size(); ++j) {
            binds[j] = gouts[0];
        }
    }
    // Inherit cublas's stream so every launch lands on the captured graph,
    // for the reason llama_like_forward_paged states at its stream setup.
    cudaStream_t stream = cublas.stream();
    // The §5.1 lora fire staging: adapters cast + grouped once per fire
    // (the hand-written `lora_state`), consumed by the HasLora guard's
    // correction launches. Constructed only when the predicate holds.
    const bool has_lora = lora != nullptr && lora->usable();
    // Campaign step 3a: prefer the ENGINE-staged state (outside any
    // capture region); local staging is the fallback.
    const LoraFireStateHandle* lora_staged =
        (has_lora && plan_state.lora_staged_table == lora)
            ? plan_state.lora_staged.get()
            : nullptr;
    std::optional<LoraFireStateHandle> lora_state;
    if (has_lora && lora_staged == nullptr) {
        lora_state.emplace(*lora, cfg, N_fire, H, Hq, Hk, I, /*tp=*/1, stream,
                           ws,
                           post_norm ? static_cast<const void*>(ws.y.data())
                                     : static_cast<const void*>(
                                           ws.norm_x.data()),
                           ws.q.data(), ws.v.data(), ws.gate.data());
    }
    if (has_lora && lora_fire_trace_enabled()) {
        std::fprintf(stderr,
                     "[lora-fire] declared R=%d lanes=%u grouping=%s%s\n",
                     R_fire, lora->count,
                     lora_staged != nullptr
                         ? lora_staged->grouping_desc().c_str()
                         : lora_state->grouping_desc().c_str(),
                     lora_staged != nullptr ? " (engine-staged)" : "");
    }

    // With padding the attention kernel runs at `dk` but the softmax must
    // stay scaled to the real head dim — `1/sqrt(d)`, the hand-written
    // override. Unpadded, -1 lets the dispatch pick `1/sqrt(dk)` (== d).
    const float sm_scale_override = head_dim_padded
        ? (1.0f / std::sqrt(static_cast<float>(d)))
        : -1.f;
    // 2c: the padded Q/K/V staging is GONE from here. The pads and the
    // strip are STATEMENTS now (`cuda::pad_head_dim` /
    // `cuda::strip_head_dim`), so their results are traced values and
    // every consumer names one -- which is why the four indirections
    // below could be four workspace fields no value described, and now
    // are none.
    //
    // `ws.attn_out` survives as the attention's fallback destination
    // for a launch whose enclosing guard names no value; see `attn_dst`.
    void* const attn_out_buf = ws.attn_out.data();

    // The HookSite slice's fire-level sidebands: the page-mask sink the
    // OnAttnProj site offers, the per-layer score captures the attention
    // publishes through, and the per-layer (possibly compacted) page
    // list the attention handlers consume. All argument-driven — an
    // unhooked fire constructs an inactive mask and none of it launches.
    model::FirePageMask page_mask(stage_hooks, stream);
    const std::uint32_t* attn_page_indices = kv_page_indices;
    const std::uint32_t* attn_page_indptr = kv_page_indptr;
    const std::uint32_t* attn_last_page_lens = kv_last_page_lens;
    std::optional<model::LayerScoreCapture> score_capture;
    std::optional<model::LayerPrefillScoreCapture> prefill_score_capture;

    // The plan caches the (unchanged) prepare hook filled. The executor no
    // longer chooses between them — each stated attention kernel's handler
    // BINDS the cache its contract needs, loudly null-checked.
    const kernels::attn::DecodePlanCache* decode_plan =
        plan_state.decode_plan ? plan_state.decode_plan.get() : nullptr;
    const kernels::attn::PrefillPlanCache* prefill_decode_plan =
        plan_state.prefill_decode_plan ? plan_state.prefill_decode_plan.get()
                                       : nullptr;
    const kernels::attn::PrefillPlanCache* prefill_plan =
        plan_state.prefill_plan ? plan_state.prefill_plan.get() : nullptr;

    const std::size_t op_count = plan.op_count();

    // A stated kernel obligates its prepare: XQA's fire-wide prepare runs
    // iff the trace states the XQA launch (a scan, not a derivation).
    for (std::size_t i = 0; i < op_count; ++i) {
        const PieForwardOp& op = plan.op(i);
        if (op.kind == PieForwardOpKind::Launch &&
            plan.weight_name(op) ==
                "attn::attention_xqa_decode_bf16_prepared") {
            kernels::attn::prepare_attention_xqa_decode_bf16(
                kv_page_indices, kv_page_indptr, kv_last_page_lens,
                R_fire, cache.page_size(), plan_state.xqa_max_pages_per_seq,
                attn_ws.view(), stream);
            break;
        }
    }

    // WHICH FACE of a row split a launch serves. Two axes, never nested
    // (the engine plans the mask split UNPLANNED for hooked fires), and
    // both are now RECTANGLE properties rather than walk state — they
    // arrive as `execute_op` parameters that these names shadow.
    //
    // `WinRegion` is the hook peel's, and only under device-window
    // capture (`peel_window_d != nullptr`): both regions launch at
    // full-N grids and the windowed call forms read the split from the
    // device word, so the captured exec is split-independent and the
    // hook fingerprint can drop the split. The marker says which face
    // of the word a site consumes (prefix = [0, w0), tail = {w0, w1}).
    //
    // `MaskRegion` is the spatial split's (NS-4 in the IR): the
    // attention call forms key their addressing on it. `None` outside
    // such peels, and inside one at its UNPLANNED endpoint — the
    // prepare kept the fire-level arm and the tail runs full-N.
    enum class WinRegion { Full, Prefix, Tail };
    enum class MaskRegion { None, Prefix, Tail };
    const bool flat_trace = flat_trace_enabled();

    // An observation SITE, as a function of one statement and its rows.
    //
    // It launches no table kernel, which is why it has no rectangle —
    // and it still runs the guest programs, stages the score capture
    // and opens the page-mask bracket, which is why a form driven by
    // the list has to run it. `Lowered::structural` names the live
    // ones; this is what running one means.
    //
    // `N` is a PARAMETER for the same reason the arms' is. A site hands
    // its programs `N` rows of the query buffer, and on a truncated
    // fire the rows past the live count at this layer are frozen at
    // whatever the last layer that owned them left behind. The list
    // carries the window (`PieForwardSite::row_lo/row_hi`); reading a
    // fire-wide count instead would show a banded fire's programs rows
    // that stopped being theirs.
    const auto execute_site = [&](const PieForwardOp& op, int N) {
        // A3: the sites live in the ONE body every unmasked fire
        // walks — a fire with no attached programs passes through
        // by argument, zero launches, zero sideband setup.
        if (stage_hooks == nullptr) return;
        const int L = static_cast<int>(op.param1);
        // What the site observes, and how wide a row of it is. Both the
        // statement's: `attn.q` names the peel's query, and the width it
        // used to spell as `Hq` is that value's.
        //
        // UNEXERCISED BY THE GATE. The parity harness runs
        // naive-baseline, which attaches no attention-stage programs --
        // `hooked=0` on all 52 fires of a run -- so `stage_hooks` is
        // null and this lambda never executes. The 168 HookSite ops in
        // the text are passed through by argument. A green gate says
        // nothing about these two lines; the fallbacks below are what
        // keeps them honest if a value is ever missing.
        const auto hook_src = [&]() -> const void* {
            const auto ins = plan.inputs(op);
            if (ins.size == 0) return ws.q.data();
            return values.slot(ins[0], plan.value(ins[0]));
        };
        const auto hook_width = [&]() {
            const auto ins = plan.inputs(op);
            if (ins.size == 0) return Hq;
            const auto& val = plan.value(ins[0]);
            std::uint32_t out = 1;
            for (std::uint32_t k = 1; k < val.rank; ++k) {
                if (val.dims[k].kind !=
                    pie_forward::PieForwardDimKind::Const) {
                    return Hq;
                }
                out *= val.dims[k].value;
            }
            return static_cast<int>(out);
        };
        if (op.param0 == 0) {
            // OnAttnProj: reset the layer's page view, re-seed the
            // mask ("keep everything" unless this layer's program
            // narrows it), stage the score capture the attention
            // will publish through, and run the programs.
            attn_page_indices = kv_page_indices;
            attn_page_indptr = kv_page_indptr;
            attn_last_page_lens = kv_last_page_lens;
            page_mask.begin_layer(stream);
            if (is_pure_decode) {
                score_capture.emplace(
                    stage_hooks, static_cast<std::uint32_t>(L),
                    static_cast<std::uint32_t>(num_q_heads),
                    /*capturable=*/true, stream);
            } else {
                prefill_score_capture.emplace(
                    stage_hooks, static_cast<std::uint32_t>(L),
                    static_cast<std::uint32_t>(num_q_heads),
                    plan_state.prefill_score_window,
                    /*capturable=*/true, stream);
            }
            // ISLAND (value arena). The observed buffer is the seam's
            // own value -- the PEEL's query, which the pin pass binds --
            // so the site stops naming `ws.q`.
            invoke_stage_hook(
                stage_hooks, StageHookPoint::OnAttnProj,
                hook_src(),
                static_cast<std::uint32_t>(N),
                static_cast<std::uint32_t>(hook_width()),
                static_cast<std::uint32_t>(L),
                stream, /*query_is_f32=*/false,
                {.mask_sink = page_mask.sink()});
        } else {
            // OnAttn: the programs read what the attention published.
            invoke_stage_hook(
                stage_hooks, StageHookPoint::OnAttn,
                hook_src(),
                static_cast<std::uint32_t>(N),
                static_cast<std::uint32_t>(hook_width()),
                static_cast<std::uint32_t>(L),
                stream, /*query_is_f32=*/false,
                {.scores = score_capture && score_capture->scores()
                               ? score_capture->scores()
                               : (prefill_score_capture
                                      ? prefill_score_capture->scores()
                                      : nullptr)});
        }
    };

    // ── THE ARMS, as a function of one RECTANGLE ───────────────────
    //
    // `.wiki/tart/dsl.md` "The cutover, sized", step 1. Everything that
    // LAUNCHES lives here; everything that TRAVERSES — the guard chain,
    // the row peels, the observation sites — stays in the walk below.
    // That is the same line `lower::Semantic::Structural` draws, and
    // drawing it in both places is what lets the second form drive
    // these arms without them noticing.
    //
    // The parameters are exactly what an arm reads about WHERE it is,
    // counted off the body before the split: a row count (N, R), a row
    // window (win_start, win_len), which face of the hook split
    // (win_region) and of the mask split (mask_region), and which
    // prepared plan (the band words). They SHADOW the walk's locals of
    // the same names, so no arm body changed — which is the argument
    // that this refactor cannot alter a launch.
    //
    // `win_region` joined the list late, and how it was missed is worth
    // recording: the count of what the arms read (the table in the wiki)
    // was taken over the words that appear in the arms' ARGUMENTS, and
    // this one appears only in their kernel-CHOICE conditions
    // (`peel_window_d != nullptr && win_region == Tail` picks the
    // `_devwin` variant). A parameter that selects a kernel is as much
    // "where am I" as one that sizes a grid.
    //
    // A `Launch{rows, layers, peel}` rectangle carries every one of
    // them, which is what step 2 will pass instead of the walk's state.
    // The epilogue's two intermediates, filled once the lowering exists
    // (below) and read by the epilogue arm. Hoisted because this
    // executor builds `flat` after the arms, not before.
    void* epi_gather = nullptr;
    void* epi_norm = nullptr;

    const auto execute_op = [&](const PieForwardOp& op,
                                std::size_t at_op,
                                int N,
                                int R,
                                int win_start,
                                int win_len,
                                WinRegion win_region,
                                MaskRegion mask_region,
                                int depth_band_index,
                                bool depth_tail_active) {
        // A value's trailing dims ARE its row width -- what `H`, `Hq`,
        // `Hk`, `I` and `V` spell. Hoisted to the arm scope because more
        // than one island wants it.
        const auto row_width = [&](std::uint32_t id) {
            const auto& val = plan.value(id);
            std::uint32_t out = 1;
            for (std::uint32_t k = 1; k < val.rank; ++k) {
                if (val.dims[k].kind !=
                    pie_forward::PieForwardDimKind::Const) {
                    return 0;
                }
                out *= val.dims[k].value;
            }
            return static_cast<int>(out);
        };
        const auto in_w = [&](std::size_t i) {
            return row_width(plan.inputs(op)[i]);
        };
        const auto out_w = [&](std::size_t i) {
            return row_width(plan.outputs(op)[i]);
        };
        // The KV write's two operands. Where the head dim is PADDED
        // they are the pad-staging buffers, which are driver scratch --
        // `pad_head_dim` has no traced destination -- so the statement's
        // values feed the pad and the staging feeds the write. Unpadded,
        // the staging IS `ws.k`/`ws.v` and the write reads the values
        // directly. qwen3-0.6b is unpadded, so the gate checks that leg
        // and not this one.
        // The attention's QUERY. Its dispatches declare no output --
        // `out []` on every one -- so only this half moves; the result
        // stays `attn_out_buf`, the same shape qwen3.5's attention has.
        //
        // What it reads is the PEEL's value, not the fused rope's: the
        // hook split is a phi over the query and the dispatches are
        // inside it. That is pinned now (see the pass), which is what
        // the load-time refusal was asking for.
        //
        // WHERE THE ATTENTION LANDS: the value the enclosing guard owns
        // -- which `o_proj` (or, on a padded deployment, the stated
        // strip) then reads by id, so the two agree without either
        // naming `ws.attn_out`. That field is the fallback for a launch
        // under no value-producing guard.
        //
        // The `head_dim_padded` arms these three carried are deleted
        // (2c): a padded fire's q, k and v ARE the pads' results, and
        // the attention's output IS the strip's operand, so every one
        // of them is an operand off the plan.
        const auto attn_dst = [&]() -> void* {
            const std::uint32_t b =
                at_op < binds.size() ? binds[at_op]
                                     : pie_forward::PIE_FORWARD_NO_VALUE;
            if (b == pie_forward::PIE_FORWARD_NO_VALUE) return attn_out_buf;
            return values.slot(b, plan.value(b));
        };
        const auto attn_src = [&]() -> const void* {
            const auto ins = plan.inputs(op);
            if (ins.size == 0) return attn_out_buf;
            return values.slot(ins[0], plan.value(ins[0]));
        };
        const auto kv_src = [&](std::size_t i, void*) -> const void* {
            return values.slot(plan.inputs(op)[i],
                               plan.value(plan.inputs(op)[i]));
        };
        switch (op.kind) {
        case PieForwardOpKind::Embed: {
            const std::string_view name = plan.weight_name(op);
            if (name != "embed") throw_unknown_weight(name);
            // ISLAND (value arena). `token_ids` stays a driver input.
            declared::arm_embed({plan, values, N, 0, stream}, op, token_ids, wb.require(name).data(), V);
            break;
        }
        case PieForwardOpKind::Rmsnorm:
            // RUNG 5: the semantic cascade is deleted -- a class
            // trace states which FOLD it runs (`cuda::rmsnorm`),
            // so this kind reaching the walk means the trace and
            // this executor drifted. Choosing here from a param
            // is what the statement now says instead.
            throw std::runtime_error(
            "declared forward: semantic Rmsnorm in a class trace "
            "(the declaration states the fold via cuda::rmsnorm)");

        case PieForwardOpKind::AddBias: {
            // Qwen-2 family qkv biases: broadcast add onto the raw
            // projection, the hand-written `maybe_add_bias` calls
            // (llama_like.cpp) argument for argument. The trace states
            // the op after the lora guard and before norms/rope, which
            // is exactly where the hand-written block sits.
            //
            // IN PLACE on its operand — `kernels::semantic_in_place`
            // joins the pair, so the arena hands both ends the same
            // bytes. That is what took the FIELD out of this arm: it
            // used to fork on q/k/v to pick a workspace buffer AND the
            // width that buffer implies, and both come off the value
            // now. What is left is the binder answering the name, which
            // is the check the fork was standing in for.
            const std::string_view name = plan.weight_name(op);
            const auto bins = plan.inputs(op);
            const auto bouts = plan.outputs(op);
            if (bins.size < 1 || bouts.size < 1) {
                throw std::runtime_error(
                    "declared forward: a bias add states " +
                    std::to_string(bins.size) + " operands and " +
                    std::to_string(bouts.size) +
                    " results, wants one of each");
            }
            const int bias_w = declared::row_width(plan, bouts[0]);
            if (bias_w <= 0) {
                throw std::runtime_error(
                    "declared forward: a bias add's result states a "
                    "non-constant row width, so the broadcast has no "
                    "extent");
            }
            kernels::norm::add_bias_bf16(
                values.slot(bouts[0], plan.value(bouts[0])),
                wb.require(name).data(), N, bias_w, stream);
            break;
        }
        case PieForwardOpKind::Matmul: {
            const std::string_view name = plan.weight_name(op);
            const ParsedWeightName nm = parse_weight_name(name);
            const auto& layer = layer_of(w, nm, name);
            const float beta = op.param0 != 0 ? 1.f : 0.f;
            // What the attention/MLP projections read: pre-norm reads the
            // normed copies (norm_x for QKV, norm_y for gate/up), post-norm
            // reads the residual stream raw — the hand-written `qkv_in` /
            // `mlp_in` indirections.
            // Every projection writes its VALUE (all of them pinned by
            // the pass, so this is the same memory the conventions named).
            const auto out_slot = [&](std::size_t i) {
                return values.slot(plan.outputs(op)[i],
                                   plan.value(plan.outputs(op)[i]));
            };

            // The island's consumers: the projection reads the value the
            // attn_norm arm produced (pinned, so LoRA's captured pointer
            // and this slot are the same bytes).
            const void* const qkv_in =
                plan.inputs(op).size > 0
                    ? static_cast<const void*>(
                          values.slot(plan.inputs(op)[0],
                                      plan.value(plan.inputs(op)[0])))
                    : (post_norm ? ws.y.data() : ws.norm_x.data());
            if (nm.field == "qkv") {
                // The packed GEMM, nothing else: whether the fused
                // decode-QKV epilogue follows is the DECLARATION's arm
                // (dsl::cuda::qkv_decode_qk_norm_rope_write_kv), stated as
                // the next Launch op — the peephole that used to re-derive
                // it here is deleted (rung 2, north-star-dsl.md).
                kernels::gemm::act_x_w(cublas.handle(),
                    qkv_in,
                    WeightView(wb.require(name)),
                    out_slot(0), N, out_w(0), in_w(0));
            } else if (nm.field == "q_proj") {
                kernels::gemm::act_x_w(cublas.handle(),
                    qkv_in,
                    dense(wb.require(name), layer.q_proj_quant, name),
                    out_slot(0), N, out_w(0), in_w(0));
            } else if (nm.field == "k_proj") {
                kernels::gemm::act_x_w(cublas.handle(),
                    qkv_in,
                    dense(wb.require(name), layer.k_proj_quant, name),
                    out_slot(0), N, out_w(0), in_w(0));
            } else if (nm.field == "v_proj") {
                kernels::gemm::act_x_w(cublas.handle(),
                    qkv_in,
                    dense(wb.require(name), layer.v_proj_quant, name),
                    out_slot(0), N, out_w(0), in_w(0));
            } else if (nm.field == "o_proj") {
                if (post_norm) {
                    // Post-norm: o_proj lands in the norm_x scratch (the
                    // trace's beta_one is 0 here); Rmsnorm(attn_norm) +
                    // ResidualAdd land it on the stream — the hand-written
                    // post-norm block, same buffers, same order.
                    kernels::gemm::act_x_w(cublas.handle(),
                        // The trace's operand order, from the builder:
                        // `matmul_inner(x, ...)` records the activation
                        // FIRST and `matmul_add` appends the residual, so
                        // inputs[0] is the activation on both forms.
                        values.slot(plan.inputs(op)[0],
                                    plan.value(plan.inputs(op)[0])),
                        dense(wb.require(name), layer.o_proj_quant, name),
                        out_slot(0), N, out_w(0), in_w(0), beta);
                } else {
                    // Residual accumulate folded into the GEMM (beta from
                    // the trace's beta_one), exactly the hand-written T==1
                    // branch.
                    kernels::gemm::act_x_w(cublas.handle(),
                        // The trace's operand order, from the builder:
                        // `matmul_inner(x, ...)` records the activation
                        // FIRST and `matmul_add` appends the residual, so
                        // inputs[0] is the activation on both forms.
                        values.slot(plan.inputs(op)[0],
                                    plan.value(plan.inputs(op)[0])),
                        dense(wb.require(name), layer.o_proj_quant, name),
                        out_slot(0), N, out_w(0), in_w(0), beta);
                }
            } else if (nm.field == "gate_up") {
                // The PACKED bank, and only that (2d). An unfused
                // binding states TWO matmuls -- each with its own weight
                // name, its own operand and its own traced result -- so
                // this branch no longer fires two GEMMs into buffers the
                // single statement did not describe, and
                // `gate_up_used_fused` goes with it.
                //
                // `require` refuses a binding that disagrees with the
                // fact the trace was taken under: a deployment without
                // the bank states the pair, so reaching this name at all
                // means the packed one.
                kernels::gemm::act_x_w(cublas.handle(),
                    values.slot(plan.inputs(op)[0],
                                plan.value(plan.inputs(op)[0])),
                    WeightView(*require(layer.gate_up_proj_fused, name)),
                    out_slot(0), N, out_w(0), in_w(0));
            } else if (nm.field == "gate_proj" || nm.field == "up_proj") {
                // One half of an unfused binding. Its result is a traced
                // value now, so it lands in the arena rather than in the
                // `ws.gate` / `ws.up` convention the single packed
                // statement forced on the activation downstream.
                kernels::gemm::act_x_w(cublas.handle(),
                    values.slot(plan.inputs(op)[0],
                                plan.value(plan.inputs(op)[0])),
                    dense(wb.require(name),
                          nm.field == "gate_proj" ? layer.gate_proj_quant
                                                  : layer.up_proj_quant,
                          name),
                    out_slot(0), N, out_w(0), in_w(0));
            } else if (nm.field == "down") {
                // The island's consumer.
                const void* const down_in =
                    values.slot(plan.inputs(op)[0],
                                plan.value(plan.inputs(op)[0]));
                if (post_norm) {
                    // Post-norm: down_proj → norm_x scratch (beta 0), then
                    // Rmsnorm(mlp_norm) + ResidualAdd — as o_proj above.
                    kernels::gemm::act_x_w(cublas.handle(),
                        down_in,
                        dense(wb.require(name), layer.down_proj_quant, name),
                        out_slot(0), N, out_w(0), in_w(0), beta);
                } else {
                    kernels::gemm::act_x_w(cublas.handle(),
                        down_in,
                        dense(wb.require(name), layer.down_proj_quant, name),
                        out_slot(0), N, out_w(0), in_w(0), beta);
                }
            } else {
                throw_unknown_weight(name);
            }
            break;
        }
        case PieForwardOpKind::SplitQkv: {
            // Windowed (A3): inside a Peel's tail region this splits the
            // hook-visible rows only, at their ABSOLUTE offsets, so the
            // full-N consumers (hooks, attention) see one contiguous
            // buffer — the hand-written tail split verbatim. Offset 0 +
            // full length is the plain full-N split.
            // ISLAND (value arena). The four buffers and the two result
            // widths are the statement's; the WINDOW is the rectangle's,
            // which is why the row offsets stay.
            void* const packed = values.slot(plan.inputs(op)[0],
                                             plan.value(plan.inputs(op)[0]));
            void* const q_out = values.slot(plan.outputs(op)[0],
                                            plan.value(plan.outputs(op)[0]));
            void* const k_out = values.slot(plan.outputs(op)[1],
                                            plan.value(plan.outputs(op)[1]));
            void* const v_out = values.slot(plan.outputs(op)[2],
                                            plan.value(plan.outputs(op)[2]));
            if (peel_window_d != nullptr && win_region == WinRegion::Tail) {
                kernels::attn::split_qkv_bf16_devwin(
                    packed, q_out, k_out, v_out,
                    peel_window_d, N, out_w(0), out_w(1), stream);
                break;
            }
            // SHARED ARM (D1). llama_like's is gemma-4's plus the row
            // WINDOW, which belongs to the rectangle and so is a
            // parameter rather than a second arm.
            declared::arm_split_qkv({plan, values, win_len, win_start, stream}, op);
            break;
        }
        case PieForwardOpKind::RmsnormPerHead: {
            // Standalone per-head norm: in place, one row per head — the
            // hand-written `rmsnorm_qk` per-head branch. (The fused
            // norm+rope kernel is no longer a peephole here: the
            // declaration's lowered arm STATES it —
            // dsl::cuda::qk_rmsnorm_rope — so a class trace never carries
            // the triple this arm used to match.)
            const std::string_view name = plan.weight_name(op);
            const ParsedWeightName nm = parse_weight_name(name);
            const auto& layer = layer_of(w, nm, name);
            // ISLAND (value arena). Two sites that differed in which
            // buffer they normed and how many HEAD-WIDE rows that is,
            // both the statement's: `op.param0` is the head width, so
            // the row count is the operand's width divided by it.
            //
            // The convention passes one pointer twice. That is the
            // convention choosing to overwrite, not the kernel needing
            // to -- it computes correctly into a fresh buffer -- so this
            // reads the operand and writes the result, and the pin pass
            // keeps both on `ws.q`/`ws.k` until rope and attention move
            // with them.
            //
            // UNEXERCISED BY THE GATE. qwen3-0.6b states the FUSED
            // `qk_rmsnorm_rope` launch instead, so no `RmsnormPerHead`
            // op reaches this walk on the model the parity harness runs
            // -- checked against its golden, which carries 84 `SplitQkv`
            // and no per-head norm. The deployments that do state it
            // standalone (`qwen3_0_6b_unfused_qkv`, gemma-4's texts) are
            // not what this family's A/B loads, so a green gate says
            // nothing about these three lines.
            if (nm.field != "q_norm" && nm.field != "k_norm") {
                throw_unknown_weight(name);
            }
            const int head = static_cast<int>(op.param0) > 0
                                 ? static_cast<int>(op.param0)
                                 : d;
            kernels::norm::rmsnorm_bf16(
                values.slot(plan.inputs(op)[0],
                            plan.value(plan.inputs(op)[0])),
                wb.require(name).data(),
                values.slot(plan.outputs(op)[0],
                            plan.value(plan.outputs(op)[0])),
                N * (in_w(0) / head), head, eps, stream);
            break;
        }
        case PieForwardOpKind::Rope:
            // RUNG 5: the width branch is deleted -- every lowered
            // trace states which rotation it runs (`cuda::rope` /
            // `cuda::rope_partial`), so the semantic kind cannot reach
            // this walk.
            throw std::runtime_error(
                "declared forward: semantic Rope in a class trace "
                "(the declaration states the rotation)");
        case PieForwardOpKind::KvAppend: {
            // RUNG 5: the write-mechanism branch is deleted — every
            // lowered trace states the KV write through the HasWriteDesc
            // guard's two launches, so the semantic kind cannot reach
            // this walk.
            throw std::runtime_error(
                "declared forward: semantic KvAppend in a class trace "
                "(the declaration states the KV write)");
        }
        case PieForwardOpKind::Attention: {
            // A class trace states its attention kernel as a Launch op;
            // the semantic kind reaching this executor means the trace
            // and the executor drifted (rung 2, north-star-dsl.md).
            throw std::runtime_error(
                "declared forward: semantic Attention op in a class trace "
                "(the declaration must state the attention kernel)");
        }
        case PieForwardOpKind::Launch: {
            // The dumb arm: resolve the STATED launcher symbol and bind.
            // Each handler is the corresponding branch of the old path
            // cascade, minus the choosing; the state layer rides param1.
            const int L = static_cast<int>(op.param1);
            // The page-mask bracket (HookSite mechanics): a written mask
            // substitutes the layer's page list into the SAME stated
            // kernel — legal only on the static (page-count-independent)
            // decode plan, the hand-written contract verbatim.
            const auto resolve_masked_pages = [&](bool takes_paged_decode) {
                if (!page_mask.written_for(static_cast<std::uint32_t>(L))) {
                    return;
                }
                if (!takes_paged_decode || decode_plan == nullptr) {
                    throw std::runtime_error(
                        "attn_page_mask was written but this layer does "
                        "not take the paged decode path");
                }
                if (!kernels::attn::decode_plan_is_page_count_independent(
                        *decode_plan)) {
                    throw std::runtime_error(
                        "attn_page_mask requires a page-count-independent "
                        "decode plan; this fire planned split-KV");
                }
                // R_fire, NOT the walk's depth-windowed `R`. The mask
                // sink is carved ONCE for the fire (`sink_.num_requests`
                // is the fire's request count), so the compaction that
                // consumes it is a fire-wide operation; handing it a
                // tail layer's live-row count made the two disagree and
                // `FirePageMask::compact` refused — which is the right
                // refusal against the wrong argument.
                //
                // Reading a prefix of the compacted CSR is exactly what a
                // narrowed layer wants: the seriation keeps live rows
                // first, and `indptr` is prefix-summed from row 0, so the
                // first `R` entries describe precisely those rows.
                page_mask.compact(
                    kv_page_indices, kv_page_indptr, kv_last_page_lens,
                    static_cast<std::uint32_t>(R_fire), stream);
                attn_page_indices = page_mask.page_indices();
                attn_page_indptr = page_mask.page_indptr();
                attn_last_page_lens = page_mask.last_page_lens();
            };
            // THE SHARED SWITCH FIRST (D1). Every symbol whose arm is
            // family-blind lives in `declared/execute.hpp` now; what
            // remains below is this family's RESIDUE -- the arms that
            // still name a workspace field, a plan cache, or a handle
            // no other family has.
            //
            // A `false` is an answer, not a failure: `resolve_kernel`
            // already refused anything the registry does not know, so
            // it means "stated, and this family executes it its own
            // way".
            const declared::ExecCtx ectx{
                {plan, values, N, win_start, stream},
                wb, cache, attn_ws, cublas, fwd_cfg.tp_comm,
                /*state_cache=*/nullptr,
                positions, qo_indptr, kv_page_indices, kv_page_indptr,
                kv_last_page_lens, row_valid_d,
                qo_indptr_h, kv_page_indptr_h,
                w_page_d, w_off_d, R,
                peel_window_d, win_region == WinRegion::Tail,
                eps, sm_scale_override, /*lse_fallback=*/nullptr,
                cfg.rope_theta,
                num_q_heads, num_kv_heads, d, dk,
                L,
                // NULL, and this is the one family for which that is a
                // decision rather than an absence.
                //
                // The other three hand their decode plan over and the
                // shared arm fires the dispatch. This family's dispatch
                // is not one call: a banded tail pairs the BAND's plan
                // with the band's own attention workspace, a union tail
                // pairs the PREFIX plan with the suffix workspace, and
                // an UnmaskedPrefix peel's regions consume hook-narrowed
                // page CSRs rather than the fire's. Three axes, and only
                // the first is a plan.
                //
                // So the plan is withheld deliberately: the shared arm
                // refuses a null and the walk falls through to the
                // residue below, which knows the other two axes. Handing
                // it a plan here would silently take the ordinary case
                // and leave the depth and mask forms unreachable.
                /*decode_plan=*/nullptr, /*prefill_plan=*/nullptr,
                /*region_dst=*/nullptr,
            };
            if (declared::execute_shared(ectx, op)) break;
            switch (declared::resolve_kernel(plan.weight_name(op))) {
            // The rope table GENERATES. It declares its result, takes
            // the fire's positions and rows and the context's head dim,
            // and refuses a zero theta the way every other rope row
            // does -- which is the whole call.
            case declared::Kernel::QkvDecodeQkNormRopeWriteKv: {
                // aux_names = [q_norm, k_norm], signature order; the
                // second INPUT, when present, is the rope-table value —
                // the trace says whether the table exists, so no latch.
                const auto aux = plan.aux_names(op);
                if (aux.size != 2) {
                    throw std::runtime_error(
                        "declared forward: fused decode-QKV launch names "
                        + std::to_string(aux.size) + " weights, wants 2");
                }
                const std::string_view q_name = plan.name(aux[0]);
                const std::string_view k_name = plan.name(aux[1]);
                const ParsedWeightName q_nm = parse_weight_name(q_name);
                if (q_nm.field != "q_norm") throw_unknown_weight(q_name);
                const auto& layer = layer_of(w, q_nm, q_name);
                const float* table =
                    plan.inputs(op).size >= 2
                        ? static_cast<const float*>(
                              values.slot(plan.inputs(op)[1],
                                          plan.value(plan.inputs(op)[1])))
                        : nullptr;
                // ISLAND (value arena), BOTH ENDS — and this one was
                // SPLIT. The packed projection is `inputs[0]`, written
                // by the `qkv` matmul above through `out_slot(0)`, and
                // this arm read `ws.qkv_fused` instead: with the host
                // assigning (the default since A2) the two are not the
                // same bytes, so the epilogue read a buffer the GEMM no
                // longer wrote. The roped q is the same story at the
                // other end — attention takes it from the arena.
                //
                // The PEEL's prefix region states no result (the peel
                // owns q; `qkv_decode_qk_norm_rope_write_kv_region`),
                // so there the destination is still the convention's,
                // and it says so rather than indexing an empty span.
                const auto fouts = plan.outputs(op);
                if (plan.inputs(op).size < 1) {
                    throw std::runtime_error(
                        "declared forward: the fused decode-QKV states no "
                        "packed operand");
                }
                const void* const packed =
                    values.slot(plan.inputs(op)[0],
                                plan.value(plan.inputs(op)[0]));
                void* const roped_q =
                    fouts.size >= 1
                        ? values.slot(fouts[0], plan.value(fouts[0]))
                        : ws.q.data();
                if (peel_window_d != nullptr) {
                    // Device-window capture: the prefix form — the word's
                    // START is this kernel's row count.
                    kernels::attn::qkv_decode_qk_norm_rope_write_kv_bf16_devwin(
                        packed,
                        roped_q,
                        cache.k(q_nm.layer), cache.v(q_nm.layer),
                        require(layer.q_norm, q_name)->data(),
                        require(layer.k_norm, k_name)->data(),
                        positions,
                        table,
                        kv_page_indices, kv_page_indptr, kv_last_page_lens,
                        has_write_desc ? w_page_d : nullptr,
                        has_write_desc ? w_off_d : nullptr,
                        row_valid_d,
                        peel_window_d,
                        N, num_q_heads, num_kv_heads, d,
                        cache.page_size(), cache.hnd_layout(),
                        cfg.rope_theta, eps, stream);
                    break;
                }
                kernels::attn::qkv_decode_qk_norm_rope_write_kv_bf16(
                    packed,
                    roped_q,
                    cache.k(q_nm.layer), cache.v(q_nm.layer),
                    require(layer.q_norm, q_name)->data(),
                    require(layer.k_norm, k_name)->data(),
                    positions,
                    table,
                    kv_page_indices, kv_page_indptr, kv_last_page_lens,
                    has_write_desc ? w_page_d : nullptr,
                    has_write_desc ? w_off_d : nullptr,
                    row_valid_d,
                    // Windowed (A3): a Peel's prefix region owns rows
                    // [0, fast_rows) — the hand-written fused call's
                    // `fast_rows` row count. Outside a peel the window
                    // is full and this is R (pure decode, N == R).
                    win_len, num_q_heads, num_kv_heads, d,
                    cache.page_size(), cache.hnd_layout(),
                    cfg.rope_theta, eps, stream);
                break;
            }
            // The fused q/k norm + rotation is SHARED. One stated
            // symbol, two launchers, and the fork is the FIRE's -- a
            // peel's tail carries a device word and the plain form does
            // not -- which is `WriteKvToPages`' shape exactly. The row
            // window it applied by hand is `ArmCtx::row` now.
            case declared::Kernel::AttentionXqaDecodePrepared: {
                resolve_masked_pages(/*takes_paged_decode=*/false);
                auto kv_view = cache.layer_view(L);
                kernels::attn::attention_xqa_decode_bf16_prepared(
                    attn_src(), kv_view.k_bf16_pages, kv_view.v_bf16_pages,
                    attn_dst(),
                    R, num_q_heads, num_kv_heads, dk,
                    cache.page_size(), plan_state.xqa_max_pages_per_seq,
                    attn_ws.view(), stream, sm_scale_override);
                break;
            }
            case declared::Kernel::AttnFlashinferDecode: {
                // STRUCTURAL S-4: tail-layer attention on a union fire
                // pairs with the PREFIX plan and its dedicated
                // workspace (the plan/workspace pairing rule).
                // ④ Act 1: a banded tail dispatch pairs the band's
                // prefix plan with the band's OWN workspace.
                if (depth_band_index >= 0 && op.depth_role == 2) {
                    const int layer_window_left_b = declared::stated_window_left(plan, op);
                    auto kv_view_b = cache.layer_view(L);
                    kernels::attn::dispatch_attention_flashinfer_decode(
                        *plan_state.depth_band_plans
                             [static_cast<std::size_t>(depth_band_index)],
                        attn_src(), kv_view_b, attn_dst(),
                        kv_page_indices, kv_page_indptr,
                        kv_last_page_lens,
                        depth_band_attn_ws_public(depth_band_index).view(),
                        stream, layer_window_left_b,
                        /*logits_soft_cap=*/0.f, sm_scale_override);
                    break;
                }
                if (depth_tail_active && op.depth_role == 2) {
                    const int layer_window_left_d = declared::stated_window_left(plan, op);
                    auto kv_view_d = cache.layer_view(L);
                    kernels::attn::dispatch_attention_flashinfer_decode(
                        *plan_state.depth_prefix_decode_plan,
                        attn_src(), kv_view_d, attn_dst(),
                        kv_page_indices, kv_page_indptr,
                        kv_last_page_lens,
                        spatial_suffix_attn_ws().view(), stream,
                        layer_window_left_d,
                        /*logits_soft_cap=*/0.f, sm_scale_override);
                    break;
                }
                if (decode_plan == nullptr) {
                    throw std::runtime_error(
                        "declared forward: trace states the flashinfer "
                        "decode kernel but prepare built no decode plan");
                }
                auto kv_view = cache.layer_view(L);
                // Same per-layer window resolution as the hand-written
                // body; runtime_window_left is -2 on this path (gate) so
                // the config-driven values decide.
                const int layer_window_left = declared::stated_window_left(plan, op);
                if (mask_region == MaskRegion::Prefix) {
                    // The UnmaskedPrefix peel's prefix region (NS-4 in
                    // the IR): the plain rows `[0, split)` against the
                    // recursively-prepared prefix decode plan. AC-4:
                    // hooked lanes ride this prefix, so the dispatch
                    // consumes the ATTN page views (hook-narrowed when
                    // sites ran; aliases of the raw CSRs otherwise).
                    resolve_masked_pages(/*takes_paged_decode=*/true);
                    kernels::attn::dispatch_attention_flashinfer_decode(
                        *decode_plan,
                        attn_src(), kv_view, attn_dst(),
                        attn_page_indices, attn_page_indptr,
                        attn_last_page_lens,
                        attn_ws.view(), stream, layer_window_left,
                        /*logits_soft_cap=*/0.f, sm_scale_override);
                    break;
                }
                resolve_masked_pages(/*takes_paged_decode=*/true);
                kernels::attn::dispatch_attention_flashinfer_decode(
                    *decode_plan,
                    attn_src(), kv_view, attn_dst(),
                    attn_page_indices, attn_page_indptr,
                    attn_last_page_lens,
                    attn_ws.view(), stream, layer_window_left,
                    /*logits_soft_cap=*/0.f, sm_scale_override);
                break;
            }
            case declared::Kernel::AttentionFlashinferDecodeCapture: {
                if (decode_plan == nullptr) {
                    throw std::runtime_error(
                        "declared forward: trace states the capture decode "
                        "kernel but prepare built no decode plan");
                }
                if (!score_capture || !score_capture->active()) {
                    throw std::runtime_error(
                        "declared forward: capture decode stated but no "
                        "active score capture (guard/pred drift)");
                }
                resolve_masked_pages(/*takes_paged_decode=*/true);
                auto kv_view = cache.layer_view(L);
                kernels::attn::dispatch_attention_flashinfer_decode_capture(
                    *decode_plan,
                    attn_src(), kv_view, attn_dst(),
                    attn_page_indices, attn_page_indptr,
                    attn_last_page_lens,
                    attn_ws.view(), stream,
                    score_capture->raw(), score_capture->indptr_d(),
                    /*window_left=*/-1,
                    /*logits_soft_cap=*/0.f, sm_scale_override);
                score_capture->publish(
                    attn_page_indptr, attn_last_page_lens,
                    cache.page_size());
                break;
            }
            case declared::Kernel::AttentionFlashinferPrefillCapture: {
                const kernels::attn::PrefillPlanCache* pp =
                    is_pure_decode ? prefill_decode_plan : prefill_plan;
                if (pp == nullptr) {
                    throw std::runtime_error(
                        "declared forward: trace states the capture "
                        "prefill kernel but prepare built no plan");
                }
                if (!prefill_score_capture ||
                    !prefill_score_capture->active()) {
                    throw std::runtime_error(
                        "declared forward: capture prefill stated but no "
                        "active score capture (guard/pred drift)");
                }
                resolve_masked_pages(/*takes_paged_decode=*/false);
                auto kv_view = cache.layer_view(L);
                kernels::attn::dispatch_attention_flashinfer_prefill_capture_bf16(
                    *pp,
                    attn_src(), kv_view.k_bf16_pages, kv_view.v_bf16_pages,
                    attn_dst(),
                    qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, attn_ws.view(), stream,
                    prefill_score_capture->raw(),
                    prefill_score_capture->folded(),
                    prefill_score_capture->indptr_d(),
                    prefill_score_capture->window(),
                    /*logits_soft_cap=*/0.f, sm_scale_override);
                prefill_score_capture->publish();
                break;
            }
            case declared::Kernel::DequantKvCacheLayerToBf16Active: {
                auto kv_view = cache.layer_view(L);
                // In a mask peel's prefix region the staging covers the
                // PLAIN lanes' pages only — beyond the split the host
                // CSR may be a composed-envelope placeholder, and the
                // suffix's custom dispatch takes the layer view whole.
                const int num_pages_in_batch =
                    mask_region == MaskRegion::Prefix
                        ? kv_page_indptr_h[plan_state.spatial_mask_split]
                        : kv_page_indptr_h[R];
                kernels::attn::dequant_kv_cache_layer_to_bf16_active(
                    kv_view, kv_page_indices, num_pages_in_batch, stream);
                break;
            }
            case declared::Kernel::AttentionFlashinferPrefillCustom: {
                // Masked PURE-DECODE fires dispatch against their
                // dedicated plan slot (the supergraph axiom): see
                // LlamaLikePlanState::mask_decode_plan.
                // The hand-written custom-mask branch, minus the choosing
                // (llama_like.cpp:1457): the custom dispatch takes the
                // layer view whole (no dequant) and the mask data rides
                // as runtime args of the stated kernel.
                const kernels::attn::PrefillPlanCache* mask_plan = is_pure_decode
                    ? (plan_state.mask_decode_plan
                           ? plan_state.mask_decode_plan.get()
                           : nullptr)
                    : prefill_plan;
                if (mask_plan == nullptr) {
                    throw std::runtime_error(
                        "declared forward: trace states the custom-mask "
                        "prefill kernel but prepare built no plan");
                }
                auto kv_view = cache.layer_view(L);
                // NS-4 in the IR: the spatial mask split is the
                // TRACE's word (the UnmaskedPrefix peel, which
                // validated planned/drift/qo-identity) — this launch
                // only spells its region's addressing. Tail = the
                // REBASED masked suffix, hybrid addressing, measured
                // live: the kernel's q/o rows are plan/qo[0]-relative
                // (offset pointers + the identity qo), the KV side
                // reads the device CSR ABSOLUTELY (base indices +
                // `+split` indptr — the composed device truth,
                // composed-envelope lanes' host views are
                // placeholders, no host rebase).
                if (mask_region == MaskRegion::Tail) {
                    // Two domains (the mixed fire): request split for
                    // CSR/mask-indptr offsets, token-row split for the
                    // q/out pointers — equal on pure-decode fires. The
                    // suffix plan lives in its DEDICATED slot for every
                    // planned tail (the mixed fire's prefill slot holds
                    // the prefix CAUSAL plan), planned against the
                    // dedicated suffix workspace.
                    const int split = plan_state.spatial_mask_split;
                    const int split_rows =
                        plan_state.spatial_mask_row_split >= 0
                            ? plan_state.spatial_mask_row_split
                            : split;
                    const kernels::attn::PrefillPlanCache* tail_plan =
                        plan_state.use_mask_decode_plan
                            ? plan_state.mask_decode_plan.get()
                            : nullptr;
                    if (tail_plan == nullptr) {
                        throw std::runtime_error(
                            "spatial mask: peel tail without a suffix "
                            "mask plan");
                    }
                    kernels::attn::dispatch_attention_flashinfer_prefill_custom(
                        *tail_plan,
                        bf16_row(attn_src(), split_rows, in_w(0)), kv_view,
                        bf16_row(attn_dst(), split_rows, Hq),
                        mask_suffix_qo_indptr_d,
                        kv_page_indices,
                        kv_page_indptr + split,
                        kv_last_page_lens + split,
                        custom_mask_d, custom_mask_indptr_d + split,
                        // Both classes now PLAN the suffix into the
                        // dedicated workspace (the pure-decode split
                        // overlaps on the side stream too).
                        spatial_suffix_attn_ws().view(),
                        stream);
                    break;
                }
                kernels::attn::dispatch_attention_flashinfer_prefill_custom(
                    *mask_plan,
                    attn_src(), kv_view, attn_dst(),
                    qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, custom_mask_d, custom_mask_indptr_d,
                    attn_ws.view(), stream);
                break;
            }
            case declared::Kernel::LoraQkvCorrection: {
                // The pseudo-symbol: one operation, many calls — the
                // hand-written apply, argument for argument (qkv_in is
                // the buffer the projections read; scratch borrows
                // ws.gate exactly as the hand-written call does). The
                // LAYER comes from the op's own tag, NOT the state
                // param (this launch addresses no implicit store, so
                // param1 rests at 0 — reading it applied layer 0's
                // adapter slice everywhere, the bug the first live A/B
                // caught).
                if (lora_staged == nullptr && !lora_state) {
                    throw std::runtime_error(
                        "declared forward: lora correction stated but no "
                        "usable lora table (guard/pred drift)");
                }
                if (op.layer < 0) {
                    throw std::runtime_error(
                        "declared forward: lora correction without a "
                        "layer tag");
                }
                // ISLAND (value arena), on the two operands it states.
                // The correction's declared inputs ARE q and v -- it
                // rewrites them where they lie, which is why it records
                // no outputs -- so those are the statement's.
                //
                // The BASE activation is not: this launch does not take
                // it as an operand, so `qkv_in` stays the convention's
                // and follows the projections' own island. The scratch
                // borrows `ws.gate` exactly as the hand-written call
                // does, and is not a value either.
                const auto lins = plan.inputs(op);
                if (lins.size < 2) {
                    throw std::runtime_error(
                        "declared forward: the lora correction states " +
                        std::to_string(lins.size) + " operands, wants q and v");
                }
                const void* const qkv_in =
                    post_norm ? ws.y.data() : ws.norm_x.data();
                (lora_staged != nullptr ? *lora_staged : *lora_state)
                    .apply(cublas.handle(), op.layer, qkv_in, H, Hq, Hk,
                           values.slot(lins[0], plan.value(lins[0])),
                           values.slot(lins[1], plan.value(lins[1])),
                           ws.gate.data());
                break;
            }
            case declared::Kernel::AttnFlashinferPrefill: {
                // Mechanical plan binding, not a choice: prepare builds
                // `prefill_plan` for prefill-shaped fires and
                // `prefill_decode_plan` for the decode-shaped
                // force_prefill fallback — the fire's shape names which
                // one this stated kernel runs against. Under
                // force_prefill_path prepare deliberately builds NO plan
                // (llama_like.cpp's early return) and the hand-written
                // body's final else runs the PLAN-LESS prefill launcher;
                // mirror it launcher for launcher (qwen2_5, the first
                // force-prefill deployment through the walk).
                if (mask_region == MaskRegion::Prefix) {
                    // The mask peel's prefix region on a force-prefill
                    // deployment: the plan-free launcher over the PLAIN
                    // rows — pure decode, so tokens == rows == split
                    // and every CSR's `[0, split]` head is the prefix's
                    // truth (the launcher reads no further).
                    const int split = plan_state.spatial_mask_split;
                    const int layer_window_left = declared::stated_window_left(plan, op);
                    auto kv_view = cache.layer_view(L);
                    kernels::attn::attention_flashinfer_prefill(
                        attn_src(), kv_view, attn_dst(),
                        qo_indptr, kv_page_indices, kv_page_indptr,
                        kv_last_page_lens,
                        qo_indptr_h, kv_page_indptr_h,
                        split, split, num_q_heads, attn_ws.view(), stream,
                        layer_window_left,
                        /*logits_soft_cap=*/0.f, sm_scale_override);
                    break;
                }
                const kernels::attn::PrefillPlanCache* pp =
                    is_pure_decode ? prefill_decode_plan : prefill_plan;
                if (pp == nullptr) {
                    if (!fwd_cfg.force_prefill_path) {
                        throw std::runtime_error(
                            "declared forward: trace states the "
                            "flashinfer prefill kernel but prepare built "
                            "no plan for this fire shape");
                    }
                    const int layer_window_left = declared::stated_window_left(plan, op);
                    auto kv_view = cache.layer_view(L);
                    kernels::attn::attention_flashinfer_prefill(
                        attn_src(), kv_view, attn_dst(),
                        qo_indptr, kv_page_indices, kv_page_indptr,
                        kv_last_page_lens,
                        qo_indptr_h, kv_page_indptr_h,
                        N, R, num_q_heads, attn_ws.view(), stream,
                        layer_window_left,
                        /*logits_soft_cap=*/0.f, sm_scale_override);
                    break;
                }
                auto kv_view = cache.layer_view(L);
                kernels::attn::dispatch_attention_flashinfer_prefill_bf16(
                    *pp,
                    attn_src(), kv_view.k_bf16_pages, kv_view.v_bf16_pages,
                    attn_dst(),
                    qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens,
                    attn_ws.view(), stream, /*logits_soft_cap=*/0.f,
                    sm_scale_override);
                // NO-DEMOTION (3-way, interpreter leg): when prepare
                // armed the middle decode plan, the causal above was
                // RE-PLANNED to the prefill lanes [0, P) — the
                // plain-decode middle [P, split_req) takes the decode
                // kernel here, exactly the hand-written pairing.
                if (mask_region == MaskRegion::Prefix &&
                    plan_state.mixed_mid_decode_plan &&
                    plan_state.mixed_mid_start >= 0) {
                    const int P = plan_state.mixed_mid_start;
                    const int mid_row =
                        static_cast<int>(qo_indptr_h[P]);
                    const int layer_window_left_m = declared::stated_window_left(plan, op);
                    kernels::attn::dispatch_attention_flashinfer_decode(
                        *plan_state.mixed_mid_decode_plan,
                        bf16_row(attn_src(), mid_row, in_w(0)), kv_view,
                        bf16_row(attn_dst(), mid_row, Hq),
                        kv_page_indices,
                        kv_page_indptr + P,
                        kv_last_page_lens + P,
                        attn_ws.view(), stream, layer_window_left_m,
                        /*logits_soft_cap=*/0.f, sm_scale_override);
                }
                break;
            }
            }
            // The attention launches land in the padded staging buffer
            // when the head dim is padded; strip before the o_proj GEMM
            // reads `[N, num_q*d]` — mechanical staging, as the pad in
            // the write handlers (the hand-written post-attention strip).
            // Keyed by NAME, not by outputs: since A1 the attention
            // launches are guard-region (output-less) forms — the guard
            // owns the value; the launch is still the buffer's writer.
            const std::string_view launch_name = plan.weight_name(op);
            const bool is_attention_out =
                launch_name == "attn::attention_xqa_decode_bf16_prepared" ||
                launch_name == "attn::dispatch_attention_flashinfer_decode" ||
                launch_name == "attn::dispatch_attention_flashinfer_prefill_bf16" ||
                launch_name == "attn::dispatch_attention_flashinfer_prefill_custom" ||
                launch_name == "attn::dispatch_attention_flashinfer_decode_capture" ||
                launch_name == "attn::dispatch_attention_flashinfer_prefill_capture_bf16";
            (void)is_attention_out;  // 2c: the strip is a statement now
            break;
        }
        case PieForwardOpKind::Swiglu: {
            // RUNG 5, again: the choice-deriving code is deleted. Every
            // lowered trace states its activation kernel (the gate_up
            // binding is a load-time fact), so the semantic kind reaching
            // this walk means the trace and the executor drifted — the
            // same refusal `Attention` and `KvAppend` make.
            throw std::runtime_error(
                "declared forward: semantic Swiglu in a class trace "
                "(the declaration states the activation kernel)");
        }
        case PieForwardOpKind::ResidualAdd: {
            // The post-norm landing: `y += norm_y` — the sub-layer's
            // normed output (Rmsnorm above wrote norm_y) accumulated onto
            // the residual stream by its own launch, exactly the
            // hand-written `kernels::norm::residual_add_bf16` calls after the
            // attn_norm and mlp_norm blocks.
            // ISLAND (value arena). `residual_add(x, residual)` lands
            // on operand 0 — the `kernel!` row aliases the result over
            // it — so the stream is the destination and the sub-layer's
            // normed output is the addend.
            declared::arm_residual_add({plan, values, N, 0, stream}, op);
            break;
        }
        case PieForwardOpKind::LmHead: {
            if (!fwd_cfg.emit_logits) break;
            const std::string_view name = plan.weight_name(op);
            // Tied embeddings trace the lm head as "embed"; either way the
            // binding already aliased `w.lm_head` accordingly.
            const DeviceTensor* lm_head =
                name == "embed" ? &wb.require(name)
                : name == "lm_head" ? &wb.require(name)
                : nullptr;
            if (lm_head == nullptr) throw_unknown_weight(name);
            // The hand-written epilogue, copied whole (T==1, no fused-AR
            // final norm on this path): compact-logit fires gather the
            // sampled rows first, then final-norm just those; full emits
            // recompute the final norm from `ws.y` for the reason
            // llama_like.cpp's comment gives (§6.2 staleness).
            //
            // This arm reads `N_fire`, not the `N` parameter, and that
            // is the whole of the epilogue's row-space wrinkle. Every
            // other op's rectangle is in `Dim::Tokens`; the epilogue's
            // is in `Dim::Requests` (`lower()` emits it over the
            // SAMPLED rows). The number this arm wants is the height of
            // `ws.y` — the fire's tokens — and the epilogue is untagged
            // by depth, so no window ever narrowed it. Saying `N_fire`
            // makes that provable at the point of use instead of
            // leaving the drive to special-case one op kind.
            const bool compact_logits =
                logit_row_indices_d != nullptr && num_logit_rows > 0 &&
                num_logit_rows < N_fire;
            // ISLAND (value arena), at the ENDS only. The stream this
            // reads and the logits it writes are the statement's; `H`
            // and `V` are their row widths.
            //
            // The two buffers BETWEEN them are not, and cannot be yet:
            // one `LmHead` lowers to a gather, a norm and a GEMM, and
            // `Lowerer::emit` hands all three the same operand run --
            // `(stream, logits)` -- so the gather's compacted hidden and
            // the normed hidden are values the flat list does not name.
            // That is D2, measured in `what_the_epilogue_hands_each_of
            // _its_rectangles`, and `ws.norm_x`/`ws.norm_y` stay until
            // the text states them.
            const auto lins = plan.inputs(op);
            const auto louts = plan.outputs(op);
            if (lins.size == 0 || louts.size == 0) {
                throw std::runtime_error(
                    "declared forward: the epilogue states no operands");
            }
            // THE INPUT STAYS `ws.y`, and the reason is the deferral
            // two arms up. The trace states `Rmsnorm(final_norm)` then
            // `LmHead` over its RESULT, but this executor defers that
            // norm -- the hand-written epilogue interleaves it with the
            // logit-row gather, and copying the block whole is what
            // keeps the two paths bit-identical -- so at this point the
            // final-norm value has not been written and its slot holds
            // the previous fire's bytes. Reading it diverged on the
            // first token.
            //
            // What this arm actually wants is that norm's OPERAND, the
            // residual stream, which belongs to an op it cannot see. So
            // the deferral costs the input its name, and says so here
            // rather than looking like an oversight.
            const void* const stream_in = ws.y.data();
            void* const logits_out =
                values.slot(louts[0], plan.value(louts[0]));
            const int hidden_w = H;
            const void* lm_head_input = nullptr;
            int lm_head_rows = N_fire;
            if (compact_logits) {
                // The lowering owns these two, not the workspace — see
                // `ValueArena::epilogue_gather`. `ws.norm_x`/`ws.norm_y`
                // were what stood in while nothing named them.
                void* const gathered = epi_gather;
                void* const normed = epi_norm;
                if (gathered == nullptr || normed == nullptr) {
                    throw std::runtime_error(
                        "declared forward: the epilogue compacts rows but "
                        "the lowering reserved no scratch for it");
                }
                kernels::layout::gather_bf16_rows(
                    static_cast<const std::uint16_t*>(stream_in),
                    logit_row_indices_d,
                    static_cast<std::uint16_t*>(gathered),
                    num_logit_rows, hidden_w, stream);
                kernels::norm::rmsnorm_bf16(
                    gathered, w.final_norm->data(),
                    normed, num_logit_rows, hidden_w, eps, stream);
                lm_head_input = normed;
                lm_head_rows = num_logit_rows;
            } else {
                void* const normed = epi_norm;
                if (normed == nullptr) {
                    throw std::runtime_error(
                        "declared forward: the epilogue norms every row but "
                        "the lowering reserved no scratch for it");
                }
                kernels::norm::rmsnorm_bf16(
                    stream_in, w.final_norm->data(), normed,
                    N_fire, hidden_w, eps, stream);
                lm_head_input = normed;
            }
            kernels::gemm::act_x_w(cublas.handle(),
                lm_head_input, WeightView(*lm_head),
                logits_out, lm_head_rows, out_w(0), hidden_w);
            break;
        }
        default:
            throw std::runtime_error(
                "declared forward: op kind " +
                std::to_string(static_cast<std::uint32_t>(op.kind)) +
                " has no emission rule");
        }
    };

    // ── WHAT A DECLARED FIRE RUNS ──────────────────────────────────
    //
    // Build the fire's rows, lower them, execute the list. That is the
    // whole of it, and it is the shape `.wiki/tart/dsl.md` asked for: a
    // loop with no vocabulary in it.
    //
    // Until `.wiki/tart/dsl.md` cutover step 3 there was a second form
    // below this one — a WALK that decided the same thing by traversing
    // the region IR, with a guard-skip stack, peel and mask index
    // events, and a depth window rebound per op. It is deleted. What
    // stood in for it while it existed was the shadow comparison, which
    // agreed on every fire for two increments before this drive was
    // written and found three real defects doing so.
    //
    // Nothing about the ARMS changed across any of it. Step 1 gave them
    // the words they read about where they are; step 2a gave the list
    // the observation sites it was missing; this maps one onto the
    // other:
    //
    //   win_start/win_len  the rectangle, directly
    //   win_region         which face of the hook split — and ONLY
    //                      under device-window capture, where the
    //                      rectangle is a full-window grid and the
    //                      split is a device word
    //   N                  the rectangle's row count, with NO
    //                      exception: the epilogue's rows are
    //                      Dim::Requests and its arm names `N_fire`
    //                      itself for the one thing it wants from token
    //                      space
    //   R                  rows == N_fire ? R_fire : rows — narrowing
    //                      makes tokens and requests the same live
    //                      count, so only the un-narrowed case
    //                      distinguishes them
    //   mask_region        which face of the spatial split
    //   band index         the band whose row count this IS. A prepared
    //                      plan is found by ROW COUNT, which is why
    //                      nothing here indexes bands and why the
    //                      three-band ceiling has nothing left to sit
    //                      on in this file (it survives in the PREPARE,
    //                      which holds at most three plans).
    {
        const std::vector<pie_forward::PieForwardRow> rows = fire_rows(
            N_fire, fast_rows, depth_k, depth_split, depth_union,
            plan_state.depth_band_k.data(), plan_state.depth_band_rows.data(),
            depth_banded ? static_cast<std::size_t>(band_count) : 0,
            (custom_mask_d != nullptr && unmasked_prefix_rows != 0xffffffffu &&
             plan_state.spatial_mask_split >= 0)
                ? plan_state.spatial_mask_split
                : -1,
            custom_mask_d != nullptr, lora != nullptr && lora->usable(),
            has_write_desc,
            stage_hooks != nullptr && stage_hooks->wants_attn_score,
            logit_row_indices_d, num_logit_rows);
        const pie_forward::PieForwardLowered flat =
            plan.lower(rows.data(), rows.size(), peel_window_d != nullptr);
        if (flat.uncovered != pie_forward::PieForwardUncovered::None) {
            throw std::runtime_error(
                "declared forward (flat): the lowering refuses this fire, "
                "reason " +
                std::to_string(static_cast<std::uint32_t>(flat.uncovered)) +
                " — an admission answer arriving too late");
        }
        // This leg HAS a lowering, so it can take the host's buffer
        // table; the op walk above cannot, and stays on pins alone. The
        // pins still win where both speak — the arms are the same arms,
        // and they have not moved yet.
        values.bind_offsets(ws.declared_values.data(),
                            ws.declared_values.nbytes(), flat);
        epi_gather = values.epilogue_gather(flat);
        epi_norm = values.epilogue_norm(flat);
        // The band a row count names. `-1` for "the whole fire", which
        // is the walk's own degenerate rule.
        const auto band_of = [&](int live) {
            if (!depth_banded || live == N_fire) return -1;
            for (int j = 0; j < band_count; ++j) {
                if (plan_state.depth_band_rows[static_cast<std::size_t>(j)] ==
                    static_cast<std::uint32_t>(live)) {
                    return j;
                }
            }
            return -1;
        };
        std::size_t next_site = 0;
        std::size_t at = 0;
        while (at < flat.launches_len || next_site < flat.structural_len) {
            // Statements run in op order, and the two lists are each in
            // that order, so this is a merge.
            const bool site_first =
                at >= flat.launches_len ||
                (next_site < flat.structural_len &&
                 flat.structural[next_site].at_op < flat.launches[at].at_op);
            if (site_first) {
                const pie_forward::PieForwardSite& S =
                    flat.structural[next_site];
                execute_site(plan.op(S.at_op),
                             static_cast<int>(S.row_hi - S.row_lo));
                ++next_site;
                continue;
            }
            const pie_forward::PieForwardLaunch& L = flat.launches[at];
            // One CALL per rectangle, not per launch: an arm that runs
            // several kernels for one statement (the epilogue's gather,
            // norm and projection) runs all of them itself, so the
            // rectangles sharing a statement and a window collapse.
            std::size_t run = at + 1;
            while (run < flat.launches_len &&
                   flat.launches[run].at_op == L.at_op &&
                   flat.launches[run].row_lo == L.row_lo &&
                   flat.launches[run].row_hi == L.row_hi &&
                   flat.launches[run].peel_axis == L.peel_axis &&
                   flat.launches[run].peel_tail == L.peel_tail) {
                ++run;
            }
            at = run;
            const PieForwardOp& op = plan.op(L.at_op);
            const int live = static_cast<int>(L.row_hi - L.row_lo);
            // The mask peel has an UNPLANNED endpoint: when the driver
            // declined the split, its tail IS the fire-level custom
            // dispatch, full-N, and the walk marks that region `None`
            // rather than `Tail`. Reading the axis alone gave the tail
            // its windowed addressing on a fire that has no suffix
            // plan — which the executor caught ("peel tail without a
            // suffix mask plan") and then a stray launch turned into an
            // illegal access. The split's existence is the word that
            // separates the two, so the drive reads it.
            const bool mask_split_planned =
                plan_state.spatial_mask_split >= 0 &&
                unmasked_prefix_rows != 0xffffffffu;
            const MaskRegion region =
                (L.peel_axis != 2 || !mask_split_planned)
                    ? MaskRegion::None
                    : L.peel_tail ? MaskRegion::Tail
                                  : MaskRegion::Prefix;
            // DEVICE-WINDOW capture (`rows_device`): the rectangle is a
            // full-window GRID and the split is a device word the
            // windowed call forms read, so the marker — not the row
            // range — is what tells a launch which face it serves. This
            // is the one thing the first drive got wrong: it read the
            // row range for every peel, which is right for a host split
            // and silently freezes THIS fire's split into a graph that
            // is supposed to replay across all of them. Byte parity
            // cannot see it (the fire it was measured on is correct);
            // only the REPLAY is wrong.
            const WinRegion window = !L.rows_device ? WinRegion::Full
                                     : L.peel_tail ? WinRegion::Tail
                                                   : WinRegion::Prefix;
            execute_op(op,
                       /*at_op=*/L.at_op,
                       /*N=*/live,
                       /*R=*/live == N_fire ? R_fire : live,
                       /*win_start=*/static_cast<int>(L.row_lo),
                       /*win_len=*/live,
                       window,
                       region,
                       band_of(live),
                       /*depth_tail_active=*/depth_union && live == depth_split);
        }
        if (flat_trace) {
            // `devwin` counts the rectangles whose rows are a GRID and
            // whose split is a device word. It is here because the
            // capture defect (step 2e) was invisible without it: a probe
            // that never takes the path passes either way, so proving a
            // fix needs the count as much as the text.
            std::size_t devwin = 0;
            for (std::size_t j = 0; j < flat.launches_len; ++j) {
                if (flat.launches[j].rows_device != 0) ++devwin;
            }
            std::fprintf(stderr,
                         "[flat] served rows=%zu launches=%zu sites=%zu devwin=%zu\n",
                         rows.size(), flat.launches_len, flat.structural_len,
                         devwin);
        }
    }
}




bool llama_like_supergraph_supported(const LlamaLikeDeclaredPlan& declared) {
    if (!generated_forward_enabled()) return false;
    return declared.facts_digest == kGeneratedDigest_qwen3_0_6b ||
           declared.facts_digest == kGeneratedDigest_olmo2_1b ||
           declared.facts_digest == kGeneratedDigest_qwen2_5_1_5b ||
           declared.facts_digest == kGeneratedDigest_mistral_7b_v03 ||
           declared.facts_digest == kGeneratedDigest_phi3_mini;
}

bool llama_like_forward_supergraph_build(
    const LlamaLikeDeclaredPlan& declared,
    const Qwen3Weights& w,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    const LlamaLikePlanState& plan_state,
    Workspace& ws,
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
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows,
    const std::uint32_t* w_page_d,
    const std::uint32_t* w_off_d,
    const std::uint8_t* row_valid_d,
    bool has_write_desc,
    const std::uint8_t* custom_mask_d,
    const std::int32_t* custom_mask_indptr_d,
    batch::SupergraphBuilder& sg) {
    if (!llama_like_supergraph_supported(declared)) return false;
    const auto run = [&](auto build_fn) {
        build_fn(w, cfg, fwd_cfg, plan_state, ws, cache, attn_ws, cublas,
                 token_ids, positions, qo_indptr, kv_page_indices,
                 kv_page_indptr, kv_last_page_lens, qo_indptr_h,
                 kv_page_indptr_h, total_tokens, num_requests,
                 logit_row_indices_d, num_logit_rows, w_page_d, w_off_d,
                 row_valid_d, has_write_desc, custom_mask_d,
                 custom_mask_indptr_d,
                 /*hooks=*/nullptr, /*lora=*/nullptr,
                 /*peel_window_d=*/nullptr,
                 /*unmasked_prefix_rows=*/0xffffffffu,
                 /*mask_suffix_qo_indptr_d=*/nullptr,
                 /*declared_max_layers=*/0xffffffffu,
                 /*declared_full_depth_rows=*/0xffffffffu, sg);
    };
    if (declared.facts_digest == kGeneratedDigest_qwen3_0_6b) {
        run(generated_llama_like_decode_qwen3_0_6b_supergraph_build);
        return true;
    }
    if (declared.facts_digest == kGeneratedDigest_olmo2_1b) {
        run(generated_llama_like_decode_olmo2_1b_supergraph_build);
        return true;
    }
    if (declared.facts_digest == kGeneratedDigest_qwen2_5_1_5b) {
        run(generated_llama_like_decode_qwen2_5_1_5b_supergraph_build);
        return true;
    }
    if (declared.facts_digest == kGeneratedDigest_mistral_7b_v03) {
        run(generated_llama_like_decode_mistral_7b_v03_supergraph_build);
        return true;
    }
    if (declared.facts_digest == kGeneratedDigest_phi3_mini) {
        run(generated_llama_like_decode_phi3_mini_supergraph_build);
        return true;
    }
    // D4: THE GENERATED BODIES ARE THIS PATH'S ONLY IMPLEMENTATION.
    //
    // Returning false is right for a deployment whose digest simply is
    // not emitted — the caller takes the plain graph, which is the
    // default anyway (`supergraph_enabled()` is off; the union was
    // RETIRED BY PROMOTION, NS-5). It is NOT right for a tree where the
    // `.inc` files have been deleted: the union would stop existing
    // while every switch still said it was available, and nothing would
    // say so.
    //
    // So a caller that ASKED for it and got nothing hears about it. The
    // silent leg stays silent, and the deliberate one cannot be removed
    // by deleting a file somewhere else.
    if (const char* v = std::getenv("PIE_SUPERGRAPH");
        v != nullptr && v[0] == '1') {
        std::fprintf(stderr,
                     "[declared-forward] PIE_SUPERGRAPH=1 but no generated "
                     "supergraph body matches digest '%s' — the union path "
                     "has no implementation outside the emitted .inc files, "
                     "so this fire takes the plain graph\n",
                     declared.facts_digest.c_str());
    }
    return false;
}

}  // namespace pie_cuda_driver::model
