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
#include "model/declared/weights.hpp"
#include "batch/supergraph.hpp"
#include "kernels/add_bias.hpp"
#include "kernels/embed.hpp"
#include "kernels/gather_rows.hpp"
#include "kernels/head_dim_pad.hpp"
#include "kernels/kv_paged.hpp"
#include "kernels/residual_add.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include "kernels/split_packed.hpp"
#include "kernels/swiglu.hpp"
#include "model/attn_page_mask.hpp"
#include "model/attn_score.hpp"
#include "model/lora.hpp"
#include "model/stage_hooks.hpp"
#include "ops/attention_flashinfer.hpp"
#include "ops/attention_xqa.hpp"
#include "ops/gemm.hpp"

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
    // The vocabulary is the TRACE's (`forward/src/dsl.rs`'s `Layer`), which
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
        case PieForwardOpKind::Rmsnorm: {
            const ParsedWeightName nm = parse_weight_name(plan.weight_name(op));
            if (nm.field == "attn_norm") {
                // LoRA's `qkv_in`, captured at fire setup.
                values.pin(outs[0], post_norm ? ws.norm_y.data()
                                              : ws.norm_x.data());
            } else if (nm.field == "final_norm") {
                values.pin(outs[0], ws.norm_y.data());
            }
            // `mlp_norm`'s output is read only by the gate/up matmul, so
            // it stays an arena value (converted island).
            break;
        }
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

// The launcher registry's vocabulary: every kernel a class trace may
// STATE as a `Launch` op (dsl::cuda's raw signatures), one enum value per
// launcher symbol. `resolve_launch_kernel` is the registry lookup; the
// executor's Launch arm switches on the result and BINDS — buffers,
// plans, staging — without choosing. A symbol outside this vocabulary
// means the trace and this executor drifted; `build` validates every
// stated symbol at model load so that drift fails at boot, not mid-fire.
enum class LaunchKernel {
    RopeStandardTable,
    QkvDecodeQkNormRopeWriteKv,
    QkRmsnormRope,
    AttentionXqaDecodePrepared,
    AttentionFlashinferDecode,
    DequantKvCacheLayerToBf16Active,
    AttentionFlashinferPrefill,
    AttentionFlashinferPrefillCustom,
    AttentionFlashinferDecodeCapture,
    AttentionFlashinferPrefillCapture,
    WriteKvExplicit,
    WriteKvToPages,
    LoraQkvCorrection,
    ChunkedSwiglu,
    Swiglu,
};

LaunchKernel resolve_launch_kernel(std::string_view kernel) {
    if (kernel == "launch_chunked_swiglu_bf16") {
        return LaunchKernel::ChunkedSwiglu;
    }
    if (kernel == "launch_swiglu_bf16") {
        return LaunchKernel::Swiglu;
    }
    if (kernel == "launch_rope_standard_table") {
        return LaunchKernel::RopeStandardTable;
    }
    if (kernel == "launch_qkv_decode_qk_norm_rope_write_kv_bf16") {
        return LaunchKernel::QkvDecodeQkNormRopeWriteKv;
    }
    if (kernel == "launch_qk_rmsnorm_rope_bf16") {
        return LaunchKernel::QkRmsnormRope;
    }
    if (kernel == "launch_attention_xqa_decode_bf16_prepared") {
        return LaunchKernel::AttentionXqaDecodePrepared;
    }
    if (kernel == "dispatch_attention_flashinfer_decode") {
        return LaunchKernel::AttentionFlashinferDecode;
    }
    if (kernel == "launch_dequant_kv_cache_layer_to_bf16_active") {
        return LaunchKernel::DequantKvCacheLayerToBf16Active;
    }
    if (kernel == "dispatch_attention_flashinfer_prefill_bf16") {
        return LaunchKernel::AttentionFlashinferPrefill;
    }
    if (kernel == "dispatch_attention_flashinfer_prefill_custom") {
        return LaunchKernel::AttentionFlashinferPrefillCustom;
    }
    if (kernel == "dispatch_attention_flashinfer_decode_capture") {
        return LaunchKernel::AttentionFlashinferDecodeCapture;
    }
    if (kernel == "dispatch_attention_flashinfer_prefill_capture_bf16") {
        return LaunchKernel::AttentionFlashinferPrefillCapture;
    }
    if (kernel == "launch_write_kv_explicit_bf16") {
        return LaunchKernel::WriteKvExplicit;
    }
    if (kernel == "launch_write_kv_to_pages") {
        return LaunchKernel::WriteKvToPages;
    }
    if (kernel == "pie_lora_qkv_correction") {
        return LaunchKernel::LoraQkvCorrection;
    }
    throw std::runtime_error(
        "declared forward: stated kernel '" + std::string(kernel) +
        "' is not in this executor's registry (the trace and the driver "
        "drifted)");
}

// Boot validation: every Launch symbol a class trace states must resolve.
void validate_stated_kernels(const pie_forward::ForwardPlan& plan) {
    const std::size_t n = plan.op_count();
    for (std::size_t i = 0; i < n; ++i) {
        const PieForwardOp& op = plan.op(i);
        if (op.kind == PieForwardOpKind::Launch) {
            (void)resolve_launch_kernel(plan.weight_name(op));
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
// Uses the helpers above (require, make_weight_view); the digest constant
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
    TensorParallel,       // all-reduces the trace does not state
    LayerBinding,         // no layers, or a count that disagrees with the config
    QuantizedProjection,  // QuantMeta WeightViews the trace does not describe
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
    case NoPlanReason::MixedFusedQkv:       return "mixed-fused-qkv";
    case NoPlanReason::QkvBiasUnbound:      return "qkv-bias-unbound";
    case NoPlanReason::QkNormConvention:    return "qk-norm-convention";
    }
    return "?";
}

}  // namespace

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
    if (fwd_cfg.tp_size > 1) {
        return refuse(NoPlanReason::TensorParallel);   // all-reduces
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
    for (const auto& layer : w.layers) {
        // Quantized projections route through QuantMeta WeightViews the
        // trace does not describe; and a mixed fused/unfused binding would
        // make the single `fused_qkv` fact a lie.
        if (layer.q_proj_quant || layer.k_proj_quant || layer.v_proj_quant ||
            layer.o_proj_quant || layer.gate_proj_quant ||
            layer.up_proj_quant || layer.down_proj_quant) {
            return refuse(NoPlanReason::QuantizedProjection);
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
        "/pad" + std::to_string(cuda.head_dim_padded);
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
        std::getenv("PIE_DECLARED_FORWARD_TRACE")) {
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
    const bool depth_banded = band_count >= 2 && depth_stated;
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
        if (std::getenv("PIE_SPATIAL_MASK_TRACE") != nullptr) {
            std::fprintf(stderr, "[depth-bands-declared] R=%d m=%d\n",
                         R_fire, band_count);
        }
    }
    // The SSA value arena (`model/declared/value_arena.hpp`): values an arm
    // asks for by id, over a workspace block. Reset once per fire; the ask
    // order is the op order, so a value keeps its address across fires of
    // the same plan — what a captured graph replays against.
    declared::ValueArena values;
    values.reset(ws.declared_values.data(), ws.declared_values.nbytes(),
                 plan, N_fire, R_fire);
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
    if (std::getenv("PIE_DECLARED_FORWARD_TRACE")) {
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
    if (has_lora && std::getenv("PIE_LORA_FIRE_TRACE") != nullptr) {
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
    // Padded Q/K/V staging (the hand-written `attn_q`/... indirection):
    // GEMM in/out buffers stay at `d`; the KV write and attention consume
    // the zero-padded `dk` copies, and the o_proj reads the stripped
    // output. All identity when the model's head_dim is a dispatch value.
    void* const attn_q = head_dim_padded ? ws.q_padded.data() : ws.q.data();
    void* const attn_k = head_dim_padded ? ws.k_padded.data() : ws.k.data();
    void* const attn_v = head_dim_padded ? ws.v_padded.data() : ws.v.data();
    void* const attn_out_buf =
        head_dim_padded ? ws.attn_out_padded.data() : ws.attn_out.data();

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
    const ops::DecodePlanCache* decode_plan =
        plan_state.decode_plan ? plan_state.decode_plan.get() : nullptr;
    const ops::PrefillPlanCache* prefill_decode_plan =
        plan_state.prefill_decode_plan ? plan_state.prefill_decode_plan.get()
                                       : nullptr;
    const ops::PrefillPlanCache* prefill_plan =
        plan_state.prefill_plan ? plan_state.prefill_plan.get() : nullptr;

    const std::size_t op_count = plan.op_count();

    // A stated kernel obligates its prepare: XQA's fire-wide prepare runs
    // iff the trace states the XQA launch (a scan, not a derivation).
    for (std::size_t i = 0; i < op_count; ++i) {
        const PieForwardOp& op = plan.op(i);
        if (op.kind == PieForwardOpKind::Launch &&
            plan.weight_name(op) ==
                "launch_attention_xqa_decode_bf16_prepared") {
            ops::prepare_attention_xqa_decode_bf16(
                kv_page_indices, kv_page_indptr, kv_last_page_lens,
                R_fire, cache.page_size(), plan_state.xqa_max_pages_per_seq,
                attn_ws, stream);
            break;
        }
    }

    // Whether the gate_up Matmul took the fused binding; decides which
    // swiglu kernel the following Swiglu op launches (the hand-written
    // `use_fused_gu` pairing).
    bool gate_up_used_fused = false;
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
            invoke_stage_hook(
                stage_hooks, StageHookPoint::OnAttnProj,
                ws.q.data(),
                static_cast<std::uint32_t>(N),
                static_cast<std::uint32_t>(Hq),
                static_cast<std::uint32_t>(L),
                stream, /*query_is_f32=*/false,
                {.mask_sink = page_mask.sink()});
        } else {
            // OnAttn: the programs read what the attention published.
            invoke_stage_hook(
                stage_hooks, StageHookPoint::OnAttn,
                ws.q.data(),
                static_cast<std::uint32_t>(N),
                static_cast<std::uint32_t>(Hq),
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
    const auto execute_op = [&](const PieForwardOp& op,
                                int N,
                                int R,
                                int win_start,
                                int win_len,
                                WinRegion win_region,
                                MaskRegion mask_region,
                                int depth_band_index,
                                bool depth_tail_active) {
        switch (op.kind) {
        case PieForwardOpKind::Embed: {
            const std::string_view name = plan.weight_name(op);
            if (name != "embed") throw_unknown_weight(name);
            kernels::launch_embed_bf16(
                token_ids, wb.require(name).data(), ws.y.data(),
                N, H, V, stream);
            break;
        }
        case PieForwardOpKind::Rmsnorm: {
            if (op.param0 !=
                static_cast<std::uint32_t>(PieForwardNormVariant::Plain)) {
                throw std::runtime_error(
                    "declared forward: only the Plain rmsnorm variant is "
                    "emitted (Gemma folding is a different arithmetic)");
            }
            const std::string_view name = plan.weight_name(op);
            const ParsedWeightName nm = parse_weight_name(name);
            if (nm.field == "attn_norm") {
                const auto& layer = layer_of(w, nm, name);
                // ISLAND (pinned value): the normed activation is a traced
                // VALUE whose buffer the pin pass states, because LoRA
                // reaches it by name outside the walk.
                void* const attn_norm_out =
                    values.slot(plan.outputs(op)[0],
                                plan.value(plan.outputs(op)[0]));
                kernels::launch_rmsnorm_bf16(
                    // Post-norm norms the o_proj OUTPUT (the pinned
                    // scratch), pre-norm the residual stream.
                    post_norm ? ws.norm_x.data() : ws.y.data(),
                    wb.require(name).data(),
                    attn_norm_out, N, H, eps, stream);
            } else if (nm.field == "mlp_norm") {
                const auto& layer = layer_of(w, nm, name);
                if (post_norm) {
                    // Post-norm: the down_proj OUTPUT (in norm_x) → norm_y.
                    kernels::launch_rmsnorm_bf16(
                        ws.norm_x.data(), wb.require(name).data(),
                        ws.norm_y.data(), N, H, eps, stream);
                } else {
                    // ISLAND (value arena): the normed activation is a
                    // TRACED VALUE, not "ws.norm_y" — that name is this
                    // family's convention, and qwen3_5 spells the same
                    // role `ws.norm_x`. Producer and consumer move
                    // together; the post-norm arm above keeps its
                    // convention until its own island moves.
                    kernels::launch_rmsnorm_bf16(
                        ws.y.data(), wb.require(name).data(),
                        values.slot(plan.outputs(op)[0],
                                    plan.value(plan.outputs(op)[0])),
                        N, H, eps, stream);
                }
            } else if (nm.field == "q_norm") {
                // Global qk-norm (olmo2): ONE row RMSNorm over the
                // flattened [N, heads * head_dim] projection, in place —
                // the hand-written `rmsnorm_qk` global branch, verbatim.
                const auto& layer = layer_of(w, nm, name);
                kernels::launch_rmsnorm_bf16(
                    ws.q.data(), wb.require(name).data(),
                    ws.q.data(), N, Hq, eps, stream);
            } else if (nm.field == "k_norm") {
                const auto& layer = layer_of(w, nm, name);
                kernels::launch_rmsnorm_bf16(
                    ws.k.data(), wb.require(name).data(),
                    ws.k.data(), N, Hk, eps, stream);
            } else if (nm.layer < 0 && nm.field == "final_norm") {
                // Deferred to LmHead: the hand-written epilogue interleaves
                // the final norm with the logit-row gather (norm is row-wise,
                // so gather-then-norm equals norm-then-gather), and copying
                // that block whole is what keeps the two paths bit-identical.
                &wb.require(name);
            } else {
                throw_unknown_weight(name);
            }
            break;
        }
        case PieForwardOpKind::AddBias: {
            // Qwen-2 family qkv biases: broadcast add onto the raw
            // projection, the hand-written `maybe_add_bias` calls
            // (llama_like.cpp) argument for argument. The trace states
            // the op after the lora guard and before norms/rope, which
            // is exactly where the hand-written block sits.
            const std::string_view name = plan.weight_name(op);
            const ParsedWeightName nm = parse_weight_name(name);
            const auto& layer = layer_of(w, nm, name);
            if (nm.field == "q_bias") {
                kernels::launch_add_bias_bf16(
                    ws.q.data(), wb.require(name).data(),
                    N, Hq, stream);
            } else if (nm.field == "k_bias") {
                kernels::launch_add_bias_bf16(
                    ws.k.data(), wb.require(name).data(),
                    N, Hk, stream);
            } else if (nm.field == "v_bias") {
                kernels::launch_add_bias_bf16(
                    ws.v.data(), wb.require(name).data(),
                    N, Hk, stream);
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
            const void* const mlp_in =
                post_norm ? ws.y.data() : ws.norm_y.data();
            if (nm.field == "qkv") {
                // The packed GEMM, nothing else: whether the fused
                // decode-QKV epilogue follows is the DECLARATION's arm
                // (dsl::cuda::qkv_decode_qk_norm_rope_write_kv), stated as
                // the next Launch op — the peephole that used to re-derive
                // it here is deleted (rung 2, north-star-dsl.md).
                ops::gemm_act_x_w(cublas.handle(),
                    qkv_in,
                    ops::WeightView(wb.require(name)),
                    out_slot(0), N, Hq + 2 * Hk, H);
            } else if (nm.field == "q_proj") {
                ops::gemm_act_x_w(cublas.handle(),
                    qkv_in,
                    make_weight_view(&wb.require(name),
                                     layer.q_proj_quant),
                    out_slot(0), N, Hq, H);
            } else if (nm.field == "k_proj") {
                ops::gemm_act_x_w(cublas.handle(),
                    qkv_in,
                    make_weight_view(&wb.require(name),
                                     layer.k_proj_quant),
                    out_slot(0), N, Hk, H);
            } else if (nm.field == "v_proj") {
                ops::gemm_act_x_w(cublas.handle(),
                    qkv_in,
                    make_weight_view(&wb.require(name),
                                     layer.v_proj_quant),
                    out_slot(0), N, Hk, H);
            } else if (nm.field == "o_proj") {
                if (post_norm) {
                    // Post-norm: o_proj lands in the norm_x scratch (the
                    // trace's beta_one is 0 here); Rmsnorm(attn_norm) +
                    // ResidualAdd land it on the stream — the hand-written
                    // post-norm block, same buffers, same order.
                    ops::gemm_act_x_w(cublas.handle(),
                        // The trace's operand order, from the builder:
                        // `matmul_inner(x, ...)` records the activation
                        // FIRST and `matmul_add` appends the residual, so
                        // inputs[0] is the activation on both forms.
                        values.slot(plan.inputs(op)[0],
                                    plan.value(plan.inputs(op)[0])),
                        make_weight_view(&wb.require(name),
                                         layer.o_proj_quant),
                        out_slot(0), N, H, Hq, beta);
                } else {
                    // Residual accumulate folded into the GEMM (beta from
                    // the trace's beta_one), exactly the hand-written T==1
                    // branch.
                    ops::gemm_act_x_w(cublas.handle(),
                        // The trace's operand order, from the builder:
                        // `matmul_inner(x, ...)` records the activation
                        // FIRST and `matmul_add` appends the residual, so
                        // inputs[0] is the activation on both forms.
                        values.slot(plan.inputs(op)[0],
                                    plan.value(plan.inputs(op)[0])),
                        make_weight_view(&wb.require(name),
                                         layer.o_proj_quant),
                        out_slot(0), N, H, Hq, beta);
                }
            } else if (nm.field == "gate_up") {
                // The island's consumer: the same traced value the
                // mlp_norm arm produced (pre-norm only — see there).
                const void* const gate_up_in =
                    post_norm ? mlp_in
                              : values.slot(plan.inputs(op)[0],
                                            plan.value(plan.inputs(op)[0]));
                // The trace declares one packed matmul either way; whether
                // the binding materialised it fused is this emitter's call,
                // the same dispatch the hand-written `use_fused_gu` makes.
                gate_up_used_fused =
                    layer.gate_up_proj_fused != nullptr &&
                    !ws.gate_up_fused.empty();
                if (gate_up_used_fused) {
                    ops::gemm_act_x_w(cublas.handle(),
                        gate_up_in,
                        ops::WeightView(*layer.gate_up_proj_fused),
                        ws.gate_up_fused.data(), N, 2 * I, H);
                } else {
                    ops::gemm_act_x_w(cublas.handle(),
                        gate_up_in,
                        make_weight_view(
                            &wb.require_field(nm.layer, "gate_proj", name),
                            layer.gate_proj_quant),
                        ws.gate.data(), N, I, H);
                    ops::gemm_act_x_w(cublas.handle(),
                        gate_up_in,
                        make_weight_view(
                            &wb.require_field(nm.layer, "up_proj", name),
                            layer.up_proj_quant),
                        ws.up.data(), N, I, H);
                }
            } else if (nm.field == "down") {
                // The island's consumer.
                const void* const down_in =
                    values.slot(plan.inputs(op)[0],
                                plan.value(plan.inputs(op)[0]));
                if (post_norm) {
                    // Post-norm: down_proj → norm_x scratch (beta 0), then
                    // Rmsnorm(mlp_norm) + ResidualAdd — as o_proj above.
                    ops::gemm_act_x_w(cublas.handle(),
                        down_in,
                        make_weight_view(&wb.require(name),
                                         layer.down_proj_quant),
                        ws.norm_x.data(), N, H, I, beta);
                } else {
                    ops::gemm_act_x_w(cublas.handle(),
                        down_in,
                        make_weight_view(&wb.require(name),
                                         layer.down_proj_quant),
                        ws.y.data(), N, H, I, beta);
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
            if (peel_window_d != nullptr && win_region == WinRegion::Tail) {
                kernels::launch_split_qkv_bf16_devwin(
                    ws.qkv_fused.data(),
                    ws.q.data(), ws.k.data(), ws.v.data(),
                    peel_window_d, N, Hq, Hk, stream);
                break;
            }
            kernels::launch_split_qkv_bf16(
                bf16_row(ws.qkv_fused.data(), win_start, Hq + 2 * Hk),
                bf16_row(ws.q.data(), win_start, Hq),
                bf16_row(ws.k.data(), win_start, Hk),
                bf16_row(ws.v.data(), win_start, Hk),
                win_len, Hq, Hk, stream);
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
            if (nm.field == "q_norm") {
                kernels::launch_rmsnorm_bf16(
                    ws.q.data(), wb.require(name).data(),
                    ws.q.data(), N * num_q_heads, d, eps, stream);
            } else if (nm.field == "k_norm") {
                kernels::launch_rmsnorm_bf16(
                    ws.k.data(), wb.require(name).data(),
                    ws.k.data(), N * num_kv_heads, d, eps, stream);
            } else {
                throw_unknown_weight(name);
            }
            break;
        }
        case PieForwardOpKind::Rope: {
            // Reached only when neither peephole consumed it (no qk-norm
            // in the trace, so the fused decode-QKV predicate — which
            // requires qk-norm — cannot hold either). Positions go straight
            // to the kernel, as the hand-written `apply_rope` does for
            // RopeKind::Standard; the rope_table serves only the fused
            // decode postprocess above.
            if (op.param0 !=
                static_cast<std::uint32_t>(PieForwardRopeKind::Standard)) {
                throw std::runtime_error(
                    "declared forward: only standard rope is emitted "
                    "(build gate admits nothing else)");
            }
            kernels::launch_rope_bf16(
                ws.q.data(), ws.k.data(), positions,
                N, num_q_heads, num_kv_heads, d,
                cfg.rope_theta, stream);
            break;
        }
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
                if (!ops::decode_plan_is_page_count_independent(
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
            switch (resolve_launch_kernel(plan.weight_name(op))) {
            // The MLP activation. The declaration states WHICH swiglu,
            // from the gate_up binding fact it was traced with, so this
            // arm no longer asks the workspace — it CHECKS, because a
            // trace built against a different binding than the one this
            // model bound would otherwise read the wrong buffer and
            // produce plausible garbage. Refuse on drift, the same rule
            // the region table's plan cross-check follows.
            case LaunchKernel::ChunkedSwiglu:
            case LaunchKernel::Swiglu: {
                const bool stated_fused =
                    resolve_launch_kernel(plan.weight_name(op)) ==
                    LaunchKernel::ChunkedSwiglu;
                if (stated_fused != gate_up_used_fused) {
                    throw std::runtime_error(
                        "declared forward: the trace states the "
                        + std::string(stated_fused ? "packed" : "unfused")
                        + " swiglu but this deployment bound the other "
                          "gate_up form (facts drift)");
                }
                void* const dst =
                    values.slot(plan.outputs(op)[0],
                                plan.value(plan.outputs(op)[0]));
                if (stated_fused) {
                    kernels::launch_chunked_swiglu_bf16(
                        ws.gate_up_fused.data(), dst, N, I, stream);
                } else {
                    kernels::launch_swiglu_bf16(
                        ws.gate.data(), ws.up.data(), dst, N * I, stream);
                }
                break;
            }
            case LaunchKernel::RopeStandardTable: {
                if (ws.rope_table.empty()) {
                    throw std::runtime_error(
                        "declared forward: trace states the rope table "
                        "build but the workspace carries no table");
                }
                kernels::launch_rope_standard_table(
                    positions,
                    static_cast<float*>(ws.rope_table.data()),
                    N, d, cfg.rope_theta, stream);
                break;
            }
            case LaunchKernel::QkvDecodeQkNormRopeWriteKv: {
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
                    plan.inputs(op).size >= 2 && !ws.rope_table.empty()
                        ? static_cast<const float*>(ws.rope_table.data())
                        : nullptr;
                if (peel_window_d != nullptr) {
                    // Device-window capture: the prefix form — the word's
                    // START is this kernel's row count.
                    kernels::launch_qkv_decode_qk_norm_rope_write_kv_bf16_devwin(
                        ws.qkv_fused.data(),
                        ws.q.data(),
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
                kernels::launch_qkv_decode_qk_norm_rope_write_kv_bf16(
                    ws.qkv_fused.data(),
                    ws.q.data(),
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
            case LaunchKernel::QkRmsnormRope: {
                const auto aux = plan.aux_names(op);
                if (aux.size != 2) {
                    throw std::runtime_error(
                        "declared forward: fused qk-norm+rope launch names "
                        + std::to_string(aux.size) + " weights, wants 2");
                }
                const std::string_view q_name = plan.name(aux[0]);
                const std::string_view k_name = plan.name(aux[1]);
                const ParsedWeightName q_nm = parse_weight_name(q_name);
                if (q_nm.field != "q_norm") throw_unknown_weight(q_name);
                const auto& layer = layer_of(w, q_nm, q_name);
                // Windowed (A3): a Peel's tail region norms+ropes the
                // hook-visible rows at their absolute offsets (the
                // hand-written tail call); offset 0 + full length is
                // the plain full-N form.
                if (peel_window_d != nullptr &&
                    win_region == WinRegion::Tail) {
                    kernels::launch_qk_rmsnorm_rope_bf16_devwin(
                        ws.q.data(), ws.k.data(),
                        require(layer.q_norm, q_name)->data(),
                        require(layer.k_norm, k_name)->data(),
                        positions,
                        peel_window_d, N,
                        num_q_heads, num_kv_heads, d,
                        cfg.rope_theta, eps, stream);
                    break;
                }
                kernels::launch_qk_rmsnorm_rope_bf16(
                    bf16_row(ws.q.data(), win_start, Hq),
                    bf16_row(ws.k.data(), win_start, Hk),
                    require(layer.q_norm, q_name)->data(),
                    require(layer.k_norm, k_name)->data(),
                    positions + win_start, win_len,
                    num_q_heads, num_kv_heads, d,
                    cfg.rope_theta, eps, stream);
                break;
            }
            case LaunchKernel::AttentionXqaDecodePrepared: {
                resolve_masked_pages(/*takes_paged_decode=*/false);
                auto kv_view = cache.layer_view(L);
                ops::launch_attention_xqa_decode_bf16_prepared(
                    attn_q, kv_view.k_bf16_pages, kv_view.v_bf16_pages,
                    attn_out_buf,
                    R, num_q_heads, num_kv_heads, dk,
                    cache.page_size(), plan_state.xqa_max_pages_per_seq,
                    attn_ws, stream, sm_scale_override);
                break;
            }
            case LaunchKernel::AttentionFlashinferDecode: {
                // STRUCTURAL S-4: tail-layer attention on a union fire
                // pairs with the PREFIX plan and its dedicated
                // workspace (the plan/workspace pairing rule).
                // ④ Act 1: a banded tail dispatch pairs the band's
                // prefix plan with the band's OWN workspace.
                if (depth_band_index >= 0 && op.depth_role == 2) {
                    const int layer_window_left_b =
                        (!fwd_cfg.per_layer_window_left.empty() &&
                         L < static_cast<int>(
                                 fwd_cfg.per_layer_window_left.size()))
                            ? fwd_cfg.per_layer_window_left[L]
                            : fwd_cfg.sliding_window;
                    auto kv_view_b = cache.layer_view(L);
                    ops::dispatch_attention_flashinfer_decode(
                        *plan_state.depth_band_plans
                             [static_cast<std::size_t>(depth_band_index)],
                        attn_q, kv_view_b, attn_out_buf,
                        kv_page_indices, kv_page_indptr,
                        kv_last_page_lens,
                        depth_band_attn_ws_public(depth_band_index),
                        stream, layer_window_left_b,
                        /*logits_soft_cap=*/0.f, sm_scale_override);
                    break;
                }
                if (depth_tail_active && op.depth_role == 2) {
                    const int layer_window_left_d =
                        (!fwd_cfg.per_layer_window_left.empty() &&
                         L < static_cast<int>(
                                 fwd_cfg.per_layer_window_left.size()))
                            ? fwd_cfg.per_layer_window_left[L]
                            : fwd_cfg.sliding_window;
                    auto kv_view_d = cache.layer_view(L);
                    ops::dispatch_attention_flashinfer_decode(
                        *plan_state.depth_prefix_decode_plan,
                        attn_q, kv_view_d, attn_out_buf,
                        kv_page_indices, kv_page_indptr,
                        kv_last_page_lens,
                        spatial_suffix_attn_ws(), stream,
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
                const int layer_window_left =
                    (!fwd_cfg.per_layer_window_left.empty() &&
                     L < static_cast<int>(
                             fwd_cfg.per_layer_window_left.size()))
                        ? fwd_cfg.per_layer_window_left[L]
                        : fwd_cfg.sliding_window;
                if (mask_region == MaskRegion::Prefix) {
                    // The UnmaskedPrefix peel's prefix region (NS-4 in
                    // the IR): the plain rows `[0, split)` against the
                    // recursively-prepared prefix decode plan. AC-4:
                    // hooked lanes ride this prefix, so the dispatch
                    // consumes the ATTN page views (hook-narrowed when
                    // sites ran; aliases of the raw CSRs otherwise).
                    resolve_masked_pages(/*takes_paged_decode=*/true);
                    ops::dispatch_attention_flashinfer_decode(
                        *decode_plan,
                        attn_q, kv_view, attn_out_buf,
                        attn_page_indices, attn_page_indptr,
                        attn_last_page_lens,
                        attn_ws, stream, layer_window_left,
                        /*logits_soft_cap=*/0.f, sm_scale_override);
                    break;
                }
                resolve_masked_pages(/*takes_paged_decode=*/true);
                ops::dispatch_attention_flashinfer_decode(
                    *decode_plan,
                    attn_q, kv_view, attn_out_buf,
                    attn_page_indices, attn_page_indptr,
                    attn_last_page_lens,
                    attn_ws, stream, layer_window_left,
                    /*logits_soft_cap=*/0.f, sm_scale_override);
                break;
            }
            case LaunchKernel::AttentionFlashinferDecodeCapture: {
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
                ops::dispatch_attention_flashinfer_decode_capture(
                    *decode_plan,
                    attn_q, kv_view, attn_out_buf,
                    attn_page_indices, attn_page_indptr,
                    attn_last_page_lens,
                    attn_ws, stream,
                    score_capture->raw(), score_capture->indptr_d(),
                    /*window_left=*/-1,
                    /*logits_soft_cap=*/0.f, sm_scale_override);
                score_capture->publish(
                    attn_page_indptr, attn_last_page_lens,
                    cache.page_size());
                break;
            }
            case LaunchKernel::AttentionFlashinferPrefillCapture: {
                const ops::PrefillPlanCache* pp =
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
                ops::dispatch_attention_flashinfer_prefill_capture_bf16(
                    *pp,
                    attn_q, kv_view.k_bf16_pages, kv_view.v_bf16_pages,
                    attn_out_buf,
                    qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, attn_ws, stream,
                    prefill_score_capture->raw(),
                    prefill_score_capture->folded(),
                    prefill_score_capture->indptr_d(),
                    prefill_score_capture->window(),
                    /*logits_soft_cap=*/0.f, sm_scale_override);
                prefill_score_capture->publish();
                break;
            }
            case LaunchKernel::DequantKvCacheLayerToBf16Active: {
                auto kv_view = cache.layer_view(L);
                // In a mask peel's prefix region the staging covers the
                // PLAIN lanes' pages only — beyond the split the host
                // CSR may be a composed-envelope placeholder, and the
                // suffix's custom dispatch takes the layer view whole.
                const int num_pages_in_batch =
                    mask_region == MaskRegion::Prefix
                        ? kv_page_indptr_h[plan_state.spatial_mask_split]
                        : kv_page_indptr_h[R];
                kernels::launch_dequant_kv_cache_layer_to_bf16_active(
                    kv_view, kv_page_indices, num_pages_in_batch, stream);
                break;
            }
            case LaunchKernel::AttentionFlashinferPrefillCustom: {
                // Masked PURE-DECODE fires dispatch against their
                // dedicated plan slot (the supergraph axiom): see
                // LlamaLikePlanState::mask_decode_plan.
                // The hand-written custom-mask branch, minus the choosing
                // (llama_like.cpp:1457): the custom dispatch takes the
                // layer view whole (no dequant) and the mask data rides
                // as runtime args of the stated kernel.
                const ops::PrefillPlanCache* mask_plan = is_pure_decode
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
                    const ops::PrefillPlanCache* tail_plan =
                        plan_state.use_mask_decode_plan
                            ? plan_state.mask_decode_plan.get()
                            : nullptr;
                    if (tail_plan == nullptr) {
                        throw std::runtime_error(
                            "spatial mask: peel tail without a suffix "
                            "mask plan");
                    }
                    ops::dispatch_attention_flashinfer_prefill_custom(
                        *tail_plan,
                        bf16_row(attn_q, split_rows, Hq), kv_view,
                        bf16_row(attn_out_buf, split_rows, Hq),
                        mask_suffix_qo_indptr_d,
                        kv_page_indices,
                        kv_page_indptr + split,
                        kv_last_page_lens + split,
                        custom_mask_d, custom_mask_indptr_d + split,
                        // Both classes now PLAN the suffix into the
                        // dedicated workspace (the pure-decode split
                        // overlaps on the side stream too).
                        spatial_suffix_attn_ws(),
                        stream);
                    break;
                }
                ops::dispatch_attention_flashinfer_prefill_custom(
                    *mask_plan,
                    attn_q, kv_view, attn_out_buf,
                    qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, custom_mask_d, custom_mask_indptr_d,
                    attn_ws, stream);
                break;
            }
            case LaunchKernel::WriteKvExplicit: {
                auto kv_view = cache.layer_view(L);
                // Mechanical pad staging for padded head dims (the
                // hand-written pre-write pad block; exactly one write
                // region runs, so this launches once per layer). A
                // windowed write never coincides with padding: the Peel
                // exists only in the fused deployment, whose facts
                // require the unpadded head dim.
                if (head_dim_padded) {
                    kernels::launch_pad_head_dim_bf16(
                        ws.q.data(), attn_q, N, num_q_heads, d, dk, stream);
                    kernels::launch_pad_head_dim_bf16(
                        ws.k.data(), attn_k, N, num_kv_heads, d, dk, stream);
                    kernels::launch_pad_head_dim_bf16(
                        ws.v.data(), attn_v, N, num_kv_heads, d, dk, stream);
                }
                // Windowed (A3): the tail rows' cells only, from their
                // slice of the descriptors — the hand-written tail
                // write; offset 0 + full length is the plain form.
                if (peel_window_d != nullptr &&
                    win_region == WinRegion::Tail) {
                    kernels::launch_write_kv_explicit_bf16_devwin(
                        kv_view, attn_k, attn_v,
                        w_page_d, w_off_d,
                        peel_window_d, N, stream, row_valid_d);
                    break;
                }
                kernels::launch_write_kv_explicit_bf16(
                    kv_view,
                    bf16_row(attn_k, win_start, Hk),
                    bf16_row(attn_v, win_start, Hk),
                    w_page_d + win_start, w_off_d + win_start,
                    win_len, stream,
                    row_valid_d != nullptr ? row_valid_d + win_start
                                           : nullptr);
                break;
            }
            case LaunchKernel::LoraQkvCorrection: {
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
                const void* const qkv_in =
                    post_norm ? ws.y.data() : ws.norm_x.data();
                (lora_staged != nullptr ? *lora_staged : *lora_state)
                    .apply(cublas.handle(), op.layer, qkv_in, H, Hq, Hk,
                           ws.q.data(), ws.v.data(), ws.gate.data());
                break;
            }
            case LaunchKernel::WriteKvToPages: {
                auto kv_view = cache.layer_view(L);
                // (Pad staging comment above applies here too.)
                if (head_dim_padded) {
                    kernels::launch_pad_head_dim_bf16(
                        ws.q.data(), attn_q, N, num_q_heads, d, dk, stream);
                    kernels::launch_pad_head_dim_bf16(
                        ws.k.data(), attn_k, N, num_kv_heads, d, dk, stream);
                    kernels::launch_pad_head_dim_bf16(
                        ws.v.data(), attn_v, N, num_kv_heads, d, dk, stream);
                }
                // Windowed (A3): base pointers stay; `first_token` skips
                // the fused-prefix rows the peel's other region already
                // wrote — the hand-written tail call verbatim (0 is the
                // plain form's default).
                if (peel_window_d != nullptr &&
                    win_region == WinRegion::Tail) {
                    kernels::launch_write_kv_to_pages_bf16_devwin(
                        kv_view, attn_k, attn_v,
                        qo_indptr, kv_page_indices, kv_page_indptr,
                        kv_last_page_lens,
                        peel_window_d, N, R, stream, row_valid_d);
                    break;
                }
                kernels::launch_write_kv_to_pages(
                    kv_view, attn_k, attn_v,
                    qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens,
                    N, R, stream, row_valid_d,
                    /*first_token=*/win_start);
                break;
            }
            case LaunchKernel::AttentionFlashinferPrefill: {
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
                    const int layer_window_left =
                        (!fwd_cfg.per_layer_window_left.empty() &&
                         L < static_cast<int>(
                                 fwd_cfg.per_layer_window_left.size()))
                            ? fwd_cfg.per_layer_window_left[L]
                            : fwd_cfg.sliding_window;
                    auto kv_view = cache.layer_view(L);
                    ops::launch_attention_flashinfer_prefill(
                        attn_q, kv_view, attn_out_buf,
                        qo_indptr, kv_page_indices, kv_page_indptr,
                        kv_last_page_lens,
                        qo_indptr_h, kv_page_indptr_h,
                        split, split, num_q_heads, attn_ws, stream,
                        layer_window_left,
                        /*logits_soft_cap=*/0.f, sm_scale_override);
                    break;
                }
                const ops::PrefillPlanCache* pp =
                    is_pure_decode ? prefill_decode_plan : prefill_plan;
                if (pp == nullptr) {
                    if (!fwd_cfg.force_prefill_path) {
                        throw std::runtime_error(
                            "declared forward: trace states the "
                            "flashinfer prefill kernel but prepare built "
                            "no plan for this fire shape");
                    }
                    const int layer_window_left =
                        (!fwd_cfg.per_layer_window_left.empty() &&
                         L < static_cast<int>(
                                 fwd_cfg.per_layer_window_left.size()))
                            ? fwd_cfg.per_layer_window_left[L]
                            : fwd_cfg.sliding_window;
                    auto kv_view = cache.layer_view(L);
                    ops::launch_attention_flashinfer_prefill(
                        attn_q, kv_view, attn_out_buf,
                        qo_indptr, kv_page_indices, kv_page_indptr,
                        kv_last_page_lens,
                        qo_indptr_h, kv_page_indptr_h,
                        N, R, num_q_heads, attn_ws, stream,
                        layer_window_left,
                        /*logits_soft_cap=*/0.f, sm_scale_override);
                    break;
                }
                auto kv_view = cache.layer_view(L);
                ops::dispatch_attention_flashinfer_prefill_bf16(
                    *pp,
                    attn_q, kv_view.k_bf16_pages, kv_view.v_bf16_pages,
                    attn_out_buf,
                    qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens,
                    attn_ws, stream, /*logits_soft_cap=*/0.f,
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
                    const int layer_window_left_m =
                        (!fwd_cfg.per_layer_window_left.empty() &&
                         L < static_cast<int>(
                                 fwd_cfg.per_layer_window_left.size()))
                            ? fwd_cfg.per_layer_window_left[L]
                            : fwd_cfg.sliding_window;
                    ops::dispatch_attention_flashinfer_decode(
                        *plan_state.mixed_mid_decode_plan,
                        bf16_row(attn_q, mid_row, Hq), kv_view,
                        bf16_row(attn_out_buf, mid_row, Hq),
                        kv_page_indices,
                        kv_page_indptr + P,
                        kv_last_page_lens + P,
                        attn_ws, stream, layer_window_left_m,
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
                launch_name == "launch_attention_xqa_decode_bf16_prepared" ||
                launch_name == "dispatch_attention_flashinfer_decode" ||
                launch_name == "dispatch_attention_flashinfer_prefill_bf16" ||
                launch_name == "dispatch_attention_flashinfer_prefill_custom" ||
                launch_name == "dispatch_attention_flashinfer_decode_capture" ||
                launch_name == "dispatch_attention_flashinfer_prefill_capture_bf16";
            if (is_attention_out && head_dim_padded) {
                kernels::launch_strip_head_dim_bf16(
                    attn_out_buf, ws.attn_out.data(),
                    N, num_q_heads, d, dk, stream);
            }
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
            // hand-written `launch_residual_add_bf16` calls after the
            // attn_norm and mlp_norm blocks.
            kernels::launch_residual_add_bf16(
                ws.y.data(), ws.norm_y.data(),
                static_cast<std::size_t>(N) * H, stream);
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
            const void* lm_head_input = nullptr;
            int lm_head_rows = N_fire;
            if (compact_logits) {
                kernels::launch_gather_bf16_rows(
                    static_cast<const std::uint16_t*>(ws.y.data()),
                    logit_row_indices_d,
                    static_cast<std::uint16_t*>(ws.norm_x.data()),
                    num_logit_rows, H, stream);
                kernels::launch_rmsnorm_bf16(
                    ws.norm_x.data(), w.final_norm->data(),
                    ws.norm_y.data(), num_logit_rows, H, eps, stream);
                lm_head_input = ws.norm_y.data();
                lm_head_rows = num_logit_rows;
            } else {
                kernels::launch_rmsnorm_bf16(
                    ws.y.data(), w.final_norm->data(), ws.norm_y.data(),
                    N_fire, H, eps, stream);
                lm_head_input = ws.norm_y.data();
            }
            ops::gemm_act_x_w(cublas.handle(),
                lm_head_input, ops::WeightView(*lm_head),
                ws.logits.data(), lm_head_rows, V, H);
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
                // The arena advances with the STATEMENT, whichever form
                // is driving: slots whose value was last read before
                // this op return to the free list, and skipping that is
                // how a bump arena runs out of block on the first fire.
                values.begin_op(S.at_op);
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
            values.begin_op(L.at_op);
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
    return false;
}

}  // namespace pie_cuda_driver::model
