#include "model/qwen3_5/declared_facts.hpp"

#include "model/qwen3_5/declared_forward.hpp"
#include "model/qwen3_5/qwen3_5_forward.hpp"
#include "model/qwen3_5/qwen3_5_moe_forward.hpp"
#include "ops/flashinfer_moe.hpp"
#include "store/recurrent_state_cache.hpp"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pie_cuda_driver::model {

namespace {

// Local mirrors of the GDN tunables (post-merge, qwen3_5_forward.cpp keeps
// its copies inside that TU's anonymous namespace — the ENV VARS are the
// shared contract, and every reader caches the same parse):
int qwen35_gdn_cached_prefill_max_tokens() {
    static const int max_tokens = [] {
        const char* v = std::getenv("PIE_QWEN35_GDN_CACHED_PREFILL_MAX_TOKENS");
        if (v == nullptr || v[0] == '\0') return 0;
        return std::max(0, std::atoi(v));
    }();
    return max_tokens;
}

int qwen35_gdn_warp_tiled_max_tokens() {
    static const int max_tokens = [] {
        const char* v = std::getenv("PIE_QWEN35_GDN_WARP_TILED_MAX_TOKENS");
        if (v == nullptr || v[0] == '\0') return 64;
        return std::max(0, std::atoi(v));
    }();
    return max_tokens;
}

bool qwen35_gdn_warp_tiled_state_persist_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("PIE_QWEN35_GDN_WARP_TILED_STATE_PERSIST");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

using pie_forward::ForwardPlan;
using pie_forward::PieForwardNormVariant;
using pie_forward::PieForwardOp;
using pie_forward::PieForwardOpKind;
using pie_forward::PieForwardQwen35HybridFacts;
using pie_forward::PIE_FORWARD_NO_LAYER;

// ── Layer-kind schedule ──────────────────────────────────────────────────

// Reduce `cfg.layer_types` to the regular full-attention interval — the
// same config-to-interval reduction the Metal driver makes
// (driver/metal/src/context.cpp: interval = first full index + 1, then the
// stack is checked against it and an irregular stack is refused instead of
// silently mis-scheduled; driver/metal/src/model/qwen3_5/geometry.hpp::
// is_full_attn is the formula the facts state). This check is the strict
// form: EVERY layer's stated kind must match the formula, not just the
// full positions. Returns -1 with `reason` set when the array is outside
// the declaration's vocabulary — the hand-written path handles those
// checkpoints alone.
int reduce_full_attn_interval(const HfConfig& cfg, std::string& reason) {
    const int L = cfg.num_hidden_layers;
    if (cfg.layer_types.empty() ||
        static_cast<int>(cfg.layer_types.size()) != L) {
        reason = "layer_types missing or wrong length";
        return -1;
    }
    int first_full = -1;
    for (int l = 0; l < L; ++l) {
        const std::string& t = cfg.layer_types[l];
        if (t != "linear_attention" && t != "full_attention") {
            reason = "unexpected layer_type '" + t + "'";
            return -1;
        }
        if (first_full < 0 && t == "full_attention") first_full = l;
    }
    if (first_full < 0) {
        // No interval to state: `full_attn_interval` places one full layer
        // at the end of every block, so an all-linear stack has no regular
        // representation (and no shipped qwen3.5 config is all-linear).
        reason = "no full_attention layer in layer_types";
        return -1;
    }
    const int interval = first_full + 1;
    for (int l = 0; l < L; ++l) {
        const bool full =
            interval <= 1 || (l % interval) == (interval - 1);
        if (full != (cfg.layer_types[l] == "full_attention")) {
            reason = "irregular layer_types schedule";
            return -1;
        }
    }
    return interval;
}

// `is_full_attn`, verbatim from the Metal geometry (and
// `Qwen35HybridFacts::is_full_attn` in forward/src/facts.rs).
bool is_full_attn_layer(int l, int interval) {
    return interval <= 1 || (l % interval) == (interval - 1);
}

// ── Weight-name resolution (the executor arc's resolver in embryo) ──────

// A plan weight name split into its layer index and field, the
// llama_like declared executor's parse (`declared_forward.cpp::
// parse_weight_name`) minus the throw: validation refuses, never errors.
struct ParsedName {
    int layer = -1;
    std::string_view field;
};

bool parse_weight_name(std::string_view name, ParsedName& out) {
    constexpr std::string_view prefix = "layer.";
    if (name.substr(0, prefix.size()) != prefix) {
        out = ParsedName{-1, name};
        return true;
    }
    const std::size_t dot = name.find('.', prefix.size());
    if (dot == std::string_view::npos) return false;
    int layer = -1;
    const char* first = name.data() + prefix.size();
    const char* last = name.data() + dot;
    const auto [ptr, ec] = std::from_chars(first, last, layer);
    if (ec != std::errc() || ptr != last || layer < 0) return false;
    out = ParsedName{layer, name.substr(dot + 1)};
    return true;
}

enum class Resolve { Yes, No, NotMine };

// The layer fields both weight structs spell identically (the la_/fa_
// banks and the block norms). A GDN field named on a full-attention layer
// (or vice versa) is unresolvable, not someone else's field.
template <class LW>
Resolve resolve_shared_layer_field(const LW& lw, std::string_view f) {
    using Kind = typename LW::Kind;
    const bool linear = lw.kind == Kind::LinearAttn;
    const auto ok = [](const void* p) {
        return p != nullptr ? Resolve::Yes : Resolve::No;
    };
    if (f == "attn_norm") return ok(lw.attn_norm_pre);
    if (f == "mlp_norm") return ok(lw.mlp_norm_pre);
    // Both layer kinds close on "o_proj" (family.rs uses the one name).
    if (f == "o_proj") {
        return ok(linear ? static_cast<const void*>(lw.la_out_proj)
                         : static_cast<const void*>(lw.fa_o_proj));
    }
    if (f == "in_proj_qkv") return linear ? ok(lw.la_in_proj_qkv) : Resolve::No;
    if (f == "in_proj_z") return linear ? ok(lw.la_in_proj_z) : Resolve::No;
    if (f == "in_proj_a") return linear ? ok(lw.la_in_proj_a) : Resolve::No;
    if (f == "in_proj_b") return linear ? ok(lw.la_in_proj_b) : Resolve::No;
    // Merge 2026-08-05: the fused GDN in-projection bank is gone upstream
    // (36c333382); a trace naming it is a stale plan, refused here.
    if (f == "in_proj_qkvz" || f == "in_proj_ba") return Resolve::No;
    if (f == "conv") return linear ? ok(lw.la_conv1d_w) : Resolve::No;
    if (f == "a_log") return linear ? ok(lw.la_A_log_fp32) : Resolve::No;
    if (f == "dt_bias") return linear ? ok(lw.la_dt_bias) : Resolve::No;
    if (f == "gate_norm") return linear ? ok(lw.la_norm_w_fp32) : Resolve::No;
    if (f == "q_proj") return !linear ? ok(lw.fa_q_proj) : Resolve::No;
    if (f == "k_proj") return !linear ? ok(lw.fa_k_proj) : Resolve::No;
    if (f == "v_proj") return !linear ? ok(lw.fa_v_proj) : Resolve::No;
    if (f == "q_norm") return !linear ? ok(lw.fa_q_norm) : Resolve::No;
    if (f == "k_norm") return !linear ? ok(lw.fa_k_norm) : Resolve::No;
    return Resolve::NotMine;
}

// Dense-arch fields. The trace writes ONE gate_up matmul; the fused bank
// vs the gate/up pair is emitter dispatch (family.rs's dense_mlp_body doc),
// so the name resolves when either spelling is bound.
bool resolve_arch_field(const Qwen3_5LayerWeights& lw, std::string_view f) {
    const bool linear = lw.kind == Qwen3_5LayerWeights::Kind::LinearAttn;
    if (f == "qgkv") return !linear && lw.fa_qgkv_proj_fused != nullptr;
    if (f == "gate_up") {
        return lw.gate_up_proj_fused != nullptr ||
               (lw.gate_proj != nullptr && lw.up_proj != nullptr);
    }
    if (f == "down") return lw.down_proj != nullptr;
    return false;
}

// MoE-arch fields. The expert templates ("expert.{e}.gate_up") resolve to
// the fused per-expert tables — `{e}` is a per-token selector the resolver
// does not expand. The shared gate_up / gate accept every bound spelling
// (plain pair, fused 2I bank, fused 2I+1 gate-carrying bank) for the same
// emitter-dispatch reason as dense gate_up.
bool resolve_arch_field(const Qwen3_5MoeLayerWeights& lw, std::string_view f) {
    if (f == "router") return lw.moe_router != nullptr;
    if (f == "expert.{e}.gate_up") return lw.moe_gate_up_proj != nullptr;
    if (f == "expert.{e}.down") return lw.moe_down_proj != nullptr;
    if (f == "shared_expert.gate_up") {
        return lw.shared_gate_up_proj != nullptr ||
               lw.shared_gate_up_gate_proj != nullptr ||
               (lw.shared_gate_proj != nullptr && lw.shared_up_proj != nullptr);
    }
    if (f == "shared_expert.down") return lw.shared_down_proj != nullptr;
    if (f == "shared_expert_gate") {
        return lw.shared_gate != nullptr ||
               lw.shared_gate_up_gate_proj != nullptr;
    }
    return false;
}

template <class W>
bool resolve_weight(const W& w, std::string_view name) {
    if (name == "embed") return w.embed != nullptr;
    if (name == "final_norm") return w.final_norm != nullptr;
    if (name == "lm_head") return w.lm_head != nullptr;
    ParsedName nm;
    if (!parse_weight_name(name, nm)) return false;
    if (nm.layer < 0 || nm.layer >= static_cast<int>(w.layers.size())) {
        return false;
    }
    const auto& lw = w.layers[nm.layer];
    switch (resolve_shared_layer_field(lw, nm.field)) {
    case Resolve::Yes: return true;
    case Resolve::No: return false;
    case Resolve::NotMine: break;
    }
    return resolve_arch_field(lw, nm.field);
}

// ── Structural validation ────────────────────────────────────────────────

// The expected (layer, kind) sequence for the facts — family.rs's emission
// order restated independently, which makes it both the op-count formula
// and the per-layer schedule check in one statement.
using Expected = std::vector<std::pair<std::int32_t, PieForwardOpKind>>;

void append_full_attn_kinds(Expected& e, std::int32_t l, bool fused_qgkv) {
    const auto push = [&](PieForwardOpKind k) { e.emplace_back(l, k); };
    push(PieForwardOpKind::Rmsnorm);
    if (fused_qgkv) {
        push(PieForwardOpKind::Matmul);   // qgkv
        push(PieForwardOpKind::SplitQkv);
    } else {
        push(PieForwardOpKind::Matmul);   // q_proj (2x wide)
        push(PieForwardOpKind::Matmul);   // k_proj
        push(PieForwardOpKind::Matmul);   // v_proj
    }
    push(PieForwardOpKind::SplitQGate);
    push(PieForwardOpKind::RmsnormPerHead);  // q_norm
    push(PieForwardOpKind::RmsnormPerHead);  // k_norm
    push(PieForwardOpKind::Rope);            // partial
    push(PieForwardOpKind::KvAppend);
    push(PieForwardOpKind::Attention);
    push(PieForwardOpKind::SigmoidGateMul);
    push(PieForwardOpKind::Matmul);          // o_proj, beta=1
}

void append_gdn_kinds(Expected& e, std::int32_t l, bool fused_in_proj) {
    const auto push = [&](PieForwardOpKind k) { e.emplace_back(l, k); };
    push(PieForwardOpKind::Rmsnorm);
    if (fused_in_proj) {
        push(PieForwardOpKind::Matmul);    // in_proj_qkvz
        push(PieForwardOpKind::SplitGdn);
        push(PieForwardOpKind::Matmul);    // in_proj_ba
        push(PieForwardOpKind::SplitGdn);
    } else {
        push(PieForwardOpKind::Matmul);    // in_proj_qkv
        push(PieForwardOpKind::Matmul);    // in_proj_z
        push(PieForwardOpKind::Matmul);    // in_proj_a
        push(PieForwardOpKind::Matmul);    // in_proj_b
    }
    push(PieForwardOpKind::CausalConv1d);
    push(PieForwardOpKind::GdnPrep);
    push(PieForwardOpKind::GatedDelta);
    push(PieForwardOpKind::RmsnormGated);
    push(PieForwardOpKind::Matmul);        // o_proj, beta=1
}

void append_dense_mlp_kinds(Expected& e, std::int32_t l) {
    const auto push = [&](PieForwardOpKind k) { e.emplace_back(l, k); };
    push(PieForwardOpKind::Rmsnorm);
    push(PieForwardOpKind::Matmul);   // gate_up
    push(PieForwardOpKind::Swiglu);
    push(PieForwardOpKind::Matmul);   // down, beta=1
}

void append_moe_mlp_kinds(Expected& e, std::int32_t l, bool shared_expert) {
    const auto push = [&](PieForwardOpKind k) { e.emplace_back(l, k); };
    push(PieForwardOpKind::Rmsnorm);
    push(PieForwardOpKind::Matmul);       // router
    push(PieForwardOpKind::TopK);
    push(PieForwardOpKind::Matmul);       // expert.{e}.gate_up
    push(PieForwardOpKind::Swiglu);
    push(PieForwardOpKind::Matmul);       // expert.{e}.down
    push(PieForwardOpKind::WeightedSum);
    if (shared_expert) {
        push(PieForwardOpKind::Matmul);   // shared_expert.gate_up
        push(PieForwardOpKind::Swiglu);
        push(PieForwardOpKind::Matmul);   // shared_expert.down
        push(PieForwardOpKind::Matmul);   // shared_expert_gate
        push(PieForwardOpKind::SigmoidGateAdd);
    }
    push(PieForwardOpKind::ResidualAdd);
}

Expected expected_sequence(const PieForwardQwen35HybridFacts& facts) {
    Expected e;
    e.emplace_back(PIE_FORWARD_NO_LAYER, PieForwardOpKind::Embed);
    for (std::uint32_t l = 0; l < facts.layers; ++l) {
        const auto li = static_cast<std::int32_t>(l);
        if (is_full_attn_layer(li, static_cast<int>(facts.full_attn_interval))) {
            append_full_attn_kinds(e, li, facts.attn.fused_qkv != 0);
        } else {
            append_gdn_kinds(e, li, facts.gdn.fused_in_proj != 0);
        }
        if (facts.mlp_is_moe != 0) {
            append_moe_mlp_kinds(e, li, facts.moe.shared_expert_intermediate > 0);
        } else {
            append_dense_mlp_kinds(e, li);
        }
    }
    e.emplace_back(PIE_FORWARD_NO_LAYER, PieForwardOpKind::Rmsnorm);  // final
    e.emplace_back(PIE_FORWARD_NO_LAYER, PieForwardOpKind::LmHead);
    return e;
}

// The structural validation: op count against the formula, per-layer kind
// sequence against the interval, and the name-resolution dry walk. Returns
// the empty string on success, else the refusal reason (already logged
// loudly where the extra context helps — the first unresolvable name).
template <class ResolveFn>
std::string validate_plan(const ForwardPlan& plan,
                          const PieForwardQwen35HybridFacts& facts,
                          ResolveFn&& resolve) {
    const Expected expected = expected_sequence(facts);
    if (plan.op_count() != expected.size()) {
        return "op count " + std::to_string(plan.op_count()) +
               " != expected " + std::to_string(expected.size());
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const PieForwardOp& op = plan.op(i);
        if (op.kind != expected[i].second || op.layer != expected[i].first) {
            return "op " + std::to_string(i) + " kind " +
                   std::to_string(static_cast<std::uint32_t>(op.kind)) +
                   "@layer " + std::to_string(op.layer) + " != expected " +
                   std::to_string(
                       static_cast<std::uint32_t>(expected[i].second)) +
                   "@layer " + std::to_string(expected[i].first);
        }
    }
    // Name-resolution dry walk: every weight name in the plan must resolve
    // against the bound weight set. GdnPrep is the one kind naming TWO
    // weights (a_log in the weight slot, dt_bias as a param0 name index —
    // pie_forward.h's op table).
    const auto check = [&](std::string_view name) -> bool {
        if (name.empty() || resolve(name)) return true;
        std::fprintf(stderr,
                     "[declared-qwen35] weight name '%.*s' does not resolve "
                     "against the bound weight set\n",
                     static_cast<int>(name.size()), name.data());
        return false;
    };
    for (std::size_t i = 0; i < plan.op_count(); ++i) {
        const PieForwardOp& op = plan.op(i);
        const std::string_view name = plan.weight_name(op);
        if (!check(name)) {
            return "weight '" + std::string(name) + "' unresolvable";
        }
        if (op.kind == PieForwardOpKind::GdnPrep) {
            const std::string_view second = plan.name(op.param0);
            if (!check(second)) {
                return "weight '" + std::string(second) + "' unresolvable";
            }
        }
    }
    return {};
}

// ── Facts extraction + the one log line ──────────────────────────────────

template <class W>
Qwen35DeclaredPlan build_impl(const HfConfig& cfg, const W& w, int tp_size) {
    constexpr bool kMoe = std::is_same_v<W, Qwen3_5MoeWeights>;
    const int L = cfg.num_hidden_layers;
    int interval = -1;

    // Refusal is a fallback, not an error (build_llama_like_declared_plan's
    // contract): log the one line with the reason and leave the plan empty
    // — the hand-written path is untouched either way.
    const auto refuse = [&](const std::string& why) {
        std::fprintf(stderr,
                     "[declared-qwen35] traced ops=0 layers=%d interval=%d "
                     "validation=refused(%s)\n",
                     L, interval, why.c_str());
        return Qwen35DeclaredPlan{};
    };

    std::string reason;
    interval = reduce_full_attn_interval(cfg, reason);
    if (interval < 0) return refuse(reason);
    // TP shards every projection and inserts all-reduces the trace has no
    // vocabulary for (llama_like's tp_size gate, same reasoning).
    if (tp_size > 1) return refuse("tp>1");
    if (w.layers.size() != static_cast<std::size_t>(L)) {
        return refuse("bound layer count != config");
    }
    if (cfg.linear_num_key_heads <= 0 || cfg.linear_num_value_heads <= 0 ||
        cfg.linear_key_head_dim <= 0 || cfg.linear_value_head_dim <= 0 ||
        cfg.linear_conv_kernel_dim <= 0) {
        return refuse("linear-attn dims unset");
    }

    // Binding facts, read from the SAME pointers the hand-written forward
    // branches on (`Lw.la_in_proj_qkvz != nullptr`, `use_fused_qgkv`) —
    // the env gates (PIE_QWEN35_FUSED_GDN_PROJ,
    // PIE_QWEN35_FUSED_FULL_ATTN_QGKV) act at bind/contract time, so the
    // pointers are their downstream truth. A mixed binding would make the
    // single per-model fact a lie; quantized projections route through
    // QuantMeta views the trace does not describe (llama_like precedent).
    bool fused_gdn = false;
    bool fused_qgkv = false;
    bool saw_linear = false;
    bool saw_full = false;
    for (int l = 0; l < L; ++l) {
        const auto& lw = w.layers[l];
        using Kind = typename std::decay_t<decltype(lw)>::Kind;
        if (lw.kind == Kind::LinearAttn) {
            // Merge 2026-08-05: upstream deleted the fused GDN input
            // projections and the switch that armed them (36c333382) —
            // the split bank is the only form the loader populates.
            const bool f = false;
            if (lw.la_in_proj_qkv == nullptr ||
                lw.la_in_proj_z == nullptr ||
                lw.la_in_proj_a == nullptr ||
                lw.la_in_proj_b == nullptr) {
                return refuse("gdn in_proj binding incomplete");
            }
            fused_gdn = f;
            saw_linear = true;
        } else {
            if (lw.fa_q_proj_quant || lw.fa_k_proj_quant ||
                lw.fa_v_proj_quant || lw.fa_o_proj_quant) {
                return refuse("quantized attention projections");
            }
            if constexpr (!kMoe) {
                const bool f = lw.fa_qgkv_proj_fused != nullptr;
                if (saw_full && f != fused_qgkv) {
                    return refuse("mixed fused/unfused qgkv binding");
                }
                fused_qgkv = f;
            }
            saw_full = true;
        }
        if constexpr (!kMoe) {
            if (lw.gate_proj_quant || lw.up_proj_quant ||
                lw.down_proj_quant) {
                return refuse("quantized mlp projections");
            }
        } else {
            if (lw.shared_gate_proj_quant || lw.shared_up_proj_quant ||
                lw.shared_down_proj_quant || lw.shared_gate_quant) {
                return refuse("quantized shared-expert projections");
            }
        }
    }

    if constexpr (kMoe) {
        // The trace's full-attention body hard-codes the 2x-wide gated q
        // (family.rs full_attn_body); the MoE forward branches on the
        // config flag, so an ungated checkpoint is outside the vocabulary.
        // (The dense forward assumes the gate unconditionally, so no gate
        // check on the dense side — trace and hand-written path agree.)
        if (!cfg.attn_output_gate) return refuse("attn_output_gate disabled");
        // The MoE block HAS a declaration — `moe_mlp_body_cuda` states the
        // fused CUTLASS leg — but this executor has no arms for it: its
        // launcher registry knows none of the MoE symbols, and
        // `qwen35_validate_stated_kernels` turns an unknown symbol into a
        // model-LOAD failure. So a plan built here would not fall back to
        // the hand-written pass, it would refuse to boot the model.
        //
        // Refusing at build is the difference between "the declaration is
        // ahead of the executor" and "this checkpoint does not run".
        // Delete this line in the commit that registers the MoE kernels.
        return refuse("the MoE block's declaration has no executor arms yet");
        if (cfg.num_experts <= 0 || cfg.num_experts_per_tok <= 0 ||
            cfg.moe_intermediate_size <= 0) {
            return refuse("moe dims unset");
        }
        // The trace gates the shared-expert block on the CONFIG width; the
        // hand-written pass gates on the bound pointers. They must agree
        // or one of the two would silently diverge from the other.
        const bool shared_cfg = cfg.shared_expert_intermediate_size > 0;
        for (int l = 0; l < L; ++l) {
            const bool shared_bound = w.layers[l].shared_down_proj != nullptr;
            if (shared_bound != shared_cfg) {
                return refuse("shared-expert config/binding disagree");
            }
        }
    }

    // The norm fold: the dense qwen3_5 forward launches
    // launch_rmsnorm_gemma_bf16 unconditionally; the MoE forward folds
    // Gemma for everything but plain qwen3_moe
    // (qwen3_5_moe_forward.cpp::uses_gemma_rmsnorm).
    PieForwardNormVariant variant = PieForwardNormVariant::Gemma;
    if constexpr (kMoe) {
        if (cfg.model_type == "qwen3_moe") variant = PieForwardNormVariant::Plain;
    }

    // The driver's own rotary derivation, verbatim
    // (qwen3_5_forward.cpp's full-attention bodies).
    const int d = cfg.head_dim;
    const int rotary_dim = std::max<int>(2,
        2 * static_cast<int>(0.5f * cfg.partial_rotary_factor * d));
    if (rotary_dim > d) return refuse("rotary_dim exceeds head_dim");

    PieForwardQwen35HybridFacts facts{};
    facts.layers = static_cast<std::uint32_t>(L);
    facts.full_attn_interval = static_cast<std::uint32_t>(interval);
    facts.vocab = static_cast<std::uint32_t>(cfg.vocab_size);
    // A binding fact: bind aliases lm_head to embed when the checkpoint
    // ties them, so pointer equality is the truth (llama_like precedent).
    facts.tied_embeddings = (w.lm_head == w.embed) ? 1 : 0;
    facts.norm_variant = static_cast<std::uint32_t>(variant);

    facts.attn.hidden = static_cast<std::uint32_t>(cfg.hidden_size);
    facts.attn.q_heads = static_cast<std::uint32_t>(cfg.num_attention_heads);
    facts.attn.kv_heads = static_cast<std::uint32_t>(cfg.num_key_value_heads);
    facts.attn.head_dim = static_cast<std::uint32_t>(d);
    facts.attn.rotary_dim = static_cast<std::uint32_t>(rotary_dim);
    facts.attn.fused_qkv = fused_qgkv ? 1 : 0;
    facts.attn.norm_variant = static_cast<std::uint32_t>(variant);

    facts.gdn.hidden = static_cast<std::uint32_t>(cfg.hidden_size);
    facts.gdn.key_heads = static_cast<std::uint32_t>(cfg.linear_num_key_heads);
    facts.gdn.value_heads =
        static_cast<std::uint32_t>(cfg.linear_num_value_heads);
    facts.gdn.key_head_dim =
        static_cast<std::uint32_t>(cfg.linear_key_head_dim);
    facts.gdn.value_head_dim =
        static_cast<std::uint32_t>(cfg.linear_value_head_dim);
    facts.gdn.conv_kernel =
        static_cast<std::uint32_t>(cfg.linear_conv_kernel_dim);
    facts.gdn.fused_in_proj = fused_gdn ? 1 : 0;
    facts.gdn.norm_variant = static_cast<std::uint32_t>(variant);

    if constexpr (kMoe) {
        facts.mlp_is_moe = 1;
        facts.moe.hidden = static_cast<std::uint32_t>(cfg.hidden_size);
        facts.moe.num_experts = static_cast<std::uint32_t>(cfg.num_experts);
        facts.moe.top_k = static_cast<std::uint32_t>(cfg.num_experts_per_tok);
        facts.moe.moe_intermediate =
            static_cast<std::uint32_t>(cfg.moe_intermediate_size);
        facts.moe.shared_expert_intermediate =
            static_cast<std::uint32_t>(cfg.shared_expert_intermediate_size);
        facts.moe.norm_variant = static_cast<std::uint32_t>(variant);
    } else {
        facts.mlp_is_moe = 0;
        facts.dense_intermediate =
            static_cast<std::uint32_t>(cfg.intermediate_size);
    }

    // Trace through the ABI wrapper. A rejected request (the entry point's
    // InvalidArgument on a malformed dimension) is a refusal here, not an
    // error — the hand-written path serves regardless.
    ForwardPlan plan;
    try {
        plan = ForwardPlan::trace_qwen3_5_hybrid(facts);
    } catch (const std::exception& e) {
        return refuse(std::string("trace failed: ") + e.what());
    }

    const std::string verdict = validate_plan(
        plan, facts,
        [&w](std::string_view name) { return resolve_weight(w, name); });
    if (!verdict.empty()) {
        std::fprintf(stderr,
                     "[declared-qwen35] traced ops=%zu layers=%d interval=%d "
                     "validation=refused(%s)\n",
                     plan.op_count(), L, interval, verdict.c_str());
        return Qwen35DeclaredPlan{};
    }

    std::fprintf(stderr,
                 "[declared-qwen35] traced ops=%zu layers=%d interval=%d "
                 "validation=OK\n",
                 plan.op_count(), L, interval);

    Qwen35DeclaredPlan out;
    out.plan = std::move(plan);
    out.fused_gdn_in_proj = fused_gdn;
    out.fused_full_attn_qgkv = fused_qgkv;
    out.full_attn_interval = interval;

    // Rung 4c-iii: the CUDA backend facts, derived ONCE from this
    // deployment — the terms the executor's hoisted booleans compute per
    // fire, computed here and handed to the declaration, whose class
    // arms STATE the kernels. Term provenance per line.
    pie_forward::PieForwardQwen35CudaFacts cuda{};
    // The recurrent-state dtype: the cache is engine-owned and invisible
    // at build, so the deployment DEFAULT stands in and the executor
    // cross-checks the live cache per fire (declared_facts.hpp).
    cuda.state_bf16 =
        RecurrentStateCache::recurrent_state_bf16_default() ? 1 : 0;
    // Warp-tiled prefill eligibility, normal-fire form: K_d <= 256 plus
    // the state-persist env gate (write_state is always true outside the
    // verify services — those are 4c-iv classes).
    cuda.warp_tiled = (cfg.linear_key_head_dim <= 256 &&
                       qwen35_gdn_warp_tiled_state_persist_enabled())
                          ? 1
                          : 0;
    cuda.warp_tiled_max =
        static_cast<std::uint32_t>(qwen35_gdn_warp_tiled_max_tokens());
    cuda.cached_max =
        static_cast<std::uint32_t>(qwen35_gdn_cached_prefill_max_tokens());
    out.cuda_state_bf16 = cuda.state_bf16 != 0;

    // The verify stash is engine-configured after model build (the MTP
    // wiring calls configure_verify_hidden_stash), so the fact is the
    // MTP deployment's normal shape — stash on — and the executor
    // cross-checks per commit fire (declared_facts.hpp).
    cuda.verify_stash = 1;
    out.cuda_verify_stash = true;

    // The dense MLP's gate_up BINDING — llama_like's reasoning verbatim
    // (declared_forward.cpp: the executor re-derived this per layer as
    // `gate_up_proj_fused != nullptr && !ws.gate_up_fused.empty()`, and
    // the second term is dead because the workspace is allocated
    // unconditionally). Layer 0 speaks for the deployment: the loader's
    // join contract accepts or declines a GROUP uniformly.
    if constexpr (!kMoe) {
        cuda.gate_up_fused =
            (!w.layers.empty() && w.layers[0].gate_up_proj_fused != nullptr) ? 1
                                                                            : 0;
    }

    // The MoE block's terms. Only the fused CUTLASS leg is stated, so
    // these say whether that leg exists and what row bound it carries;
    // the trace refuses the block outright when any of them says no,
    // and the fire declines above the bound.
    if constexpr (kMoe) {
        // 512 rather than `min(max_tokens, 512)`: the workspace is sized
        // for `min(max_tokens, 512)` rows and no fire carries more than
        // `max_tokens`, so the smaller term never binds. That is what
        // lets this be derived here, where `max_tokens` is not in scope.
        //
        // The env gate is NOT the condition. The forward reads
        // `!cutlass_ws.empty()`, and the workspace is empty whenever the
        // SIZE QUERY reports zero — which it does on any arch whose
        // grouped-GEMM units this build did not compile (sm90 today: the
        // vendored units are sm80 and, behind PIE_HAS_SM100, sm100). So
        // ask the same question the forward asks. Mirroring the env
        // instead would declare a fused leg on exactly the machines that
        // fall back to the unfused path.
        //
        // The query is also the arch probe, and in this tree it still
        // THROWS rather than reporting zero when no config is backed
        // (upstream 48c280d45 turns that into a zero). Catching here
        // gives the same answer without waiting for the merge, and a
        // throw means the same thing zero does: no fused leg.
        cuda.moe_cutlass_max_rows = 0;
        if (ops::flashinfer_cutlass_moe_enabled()) {
            std::size_t bytes = 0;
            try {
                bytes = ops::flashinfer_cutlass_moe_workspace_bytes(
                    ops::MoeActivation::Swiglu, 512, cfg.hidden_size,
                    cfg.moe_intermediate_size, cfg.num_experts,
                    cfg.num_experts_per_tok, /*tp_size=*/1, /*tp_rank=*/0);
            } catch (const std::exception&) {
                bytes = 0;
            }
            cuda.moe_cutlass_max_rows = (bytes > 0) ? 512u : 0u;
        }
        // `add_to_residual` is `(T == 1) && use_decode_fast_path`; the
        // tp term is the deployment's, the other is the class's.
        cuda.moe_residual_fold = (tp_size == 1) ? 1 : 0;
        cuda.moe_force_general = qwen35_moe_force_general_path() ? 1 : 0;
        // Streamed experts are a per-layer binding, but the pass reads
        // one flag for the whole block, so disagreement between layers
        // would already be a load bug. Any layer paging its experts
        // takes the whole model off the device-side legs.
        bool streamed = false;
        bool shared_gate_dot = true;
        for (const auto& lw : w.layers) {
            if (lw.expert_cache != nullptr) streamed = true;
            // The fused dot landing needs the gate bound and unquantized;
            // a checkpoint with no shared expert never reads it, and the
            // trace only consults this fact when it has one.
            if (lw.shared_gate == nullptr || lw.shared_gate_quant.has_value()) {
                shared_gate_dot = false;
            }
        }
        cuda.moe_streamed_experts = streamed ? 1 : 0;
        cuda.moe_shared_gate_dot = shared_gate_dot ? 1 : 0;
    }

    // The digest naming what these traces were taken from — one format,
    // two printers (this and `emit_qwen35::facts_digest`); the live
    // static-form gate is what holds them together, llama's mechanism.
    out.facts_digest =
        "qwen3_5/l" + std::to_string(facts.layers) +
        "/int" + std::to_string(facts.full_attn_interval) +
        "/v" + std::to_string(facts.vocab) +
        "/te" + std::to_string(facts.tied_embeddings) +
        "/nv" + std::to_string(facts.norm_variant) +
        "/ah" + std::to_string(facts.attn.hidden) +
        "/aqh" + std::to_string(facts.attn.q_heads) +
        "/akvh" + std::to_string(facts.attn.kv_heads) +
        "/ahd" + std::to_string(facts.attn.head_dim) +
        "/arot" + std::to_string(facts.attn.rotary_dim) +
        "/afq" + std::to_string(facts.attn.fused_qkv) +
        "/gkh" + std::to_string(facts.gdn.key_heads) +
        "/gvh" + std::to_string(facts.gdn.value_heads) +
        "/gkd" + std::to_string(facts.gdn.key_head_dim) +
        "/gvd" + std::to_string(facts.gdn.value_head_dim) +
        "/gck" + std::to_string(facts.gdn.conv_kernel) +
        "/gfi" + std::to_string(facts.gdn.fused_in_proj) +
        "/moe" + std::to_string(facts.mlp_is_moe) +
        "/di" + std::to_string(facts.dense_intermediate) +
        "/sb" + std::to_string(cuda.state_bf16) +
        "/wt" + std::to_string(cuda.warp_tiled) +
        "/wtm" + std::to_string(cuda.warp_tiled_max) +
        "/cm" + std::to_string(cuda.cached_max) +
        "/vs" + std::to_string(cuda.verify_stash);

    out.decode = pie_forward::ForwardPlan::trace_qwen3_5_hybrid_cuda(
        facts, cuda, pie_forward::PieForwardFireClass::Decode);
    out.prefill = pie_forward::ForwardPlan::trace_qwen3_5_hybrid_cuda(
        facts, cuda, pie_forward::PieForwardFireClass::Prefill);
    out.commit_advance = pie_forward::ForwardPlan::trace_qwen3_5_hybrid_cuda(
        facts, cuda, pie_forward::PieForwardFireClass::CommitAdvance);
    out.state_only = pie_forward::ForwardPlan::trace_qwen3_5_hybrid_cuda(
        facts, cuda, pie_forward::PieForwardFireClass::StateOnly);
    out.frozen_verify = pie_forward::ForwardPlan::trace_qwen3_5_hybrid_cuda(
        facts, cuda, pie_forward::PieForwardFireClass::FrozenVerify);
    // Drift between the declaration's stated kernels and the executor's
    // registry fails at model load, not mid-fire.
    qwen35_validate_stated_kernels(out.decode);
    qwen35_validate_stated_kernels(out.prefill);
    qwen35_validate_stated_kernels(out.commit_advance);
    qwen35_validate_stated_kernels(out.state_only);
    qwen35_validate_stated_kernels(out.frozen_verify);
    return out;
}

}  // namespace

// SAME env name as llama_like's gate, DELIBERATELY OPPOSITE DEFAULT,
// and that is worth a warning rather than a tidy-up.
//
// llama_like flipped to default-on at cutover step 4(a). This family did
// not, because it cannot currently be validated end to end: qwen3.5 CUDA
// serving emits garbage at the upstream dev head (reproduced
// byte-identically on pure dev; `.wiki/tart/upstream_findings.md` entry
// 5), so a default-on declared path here would be an unmeasured path on
// by default. It goes on when that is fixed and the family's own parity
// bar runs green — not before, and not for symmetry.
//
// The consequence to remember: an unset `PIE_DECLARED_FORWARD` means
// DECLARED for llama_like and HAND-WRITTEN here, so any test reading the
// env to label its run must know which family it is testing
// (`cuda_gdn_site_summary_parity` reads it for this one).
bool qwen35_declared_forward_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_DECLARED_FORWARD");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

Qwen35DeclaredPlan build_qwen3_5_declared_plan(
    const HfConfig& cfg, const Qwen3_5Weights& w, int tp_size) {
    return build_impl(cfg, w, tp_size);
}

Qwen35DeclaredPlan build_qwen3_5_declared_plan(
    const HfConfig& cfg, const Qwen3_5MoeWeights& w, int tp_size) {
    return build_impl(cfg, w, tp_size);
}

}  // namespace pie_cuda_driver::model
