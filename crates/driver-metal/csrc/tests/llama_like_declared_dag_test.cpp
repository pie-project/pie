// The llama_like declared-dag emitter, checked with no GPU.
//
// North-star #5's first increment: a second backend (this one) implements one
// emitter over the declared forward plan. Pure geometry + DAG shape, the same
// bargain gemma4_decode_step_test strikes — no Metal, no checkpoint, no GPU —
// and deliberately NOT linked against pie_driver_metal_lib: the descriptor
// layer is a header and the plan comes from the pie-forward staticlib.
//
// Two plans are exercised, both traced through the C ABI
// (`pie_forward_trace_llama_like`) and each cross-checked op for op against
// the committed golden JSON (`forward/tests/golden/*.json`) so the ABI plan
// this test feeds the emitter is pinned to the artifact the forward crate
// reviews diffs of:
//
//   * qwen3_0_6b_unfused_qkv.json — unfused binding, qk-norm ON
//   * phi3_mini.json              — unfused binding, qk-norm OFF
//     (phi3's golden IS the unfused form: the checkpoint ships qkv fused but
//     the loader contract splits it into banded views, so the deployment
//     binds three projections — see LlamaLikeFacts::phi3_mini's comment.)

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "batch/scratch_color.hpp"
#include "model/llama_like/declared_dag.hpp"
#include "pie_forward/plan.hpp"

using namespace pie::metal::llama_like;
using json = nlohmann::json;

namespace {

int g_failures = 0;

bool expect(bool ok, const std::string& what) {
    std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_failures;
    return ok;
}

bool expect_eq(long long got, long long want, const std::string& what) {
    return expect(got == want,
                  what + " (got " + std::to_string(got) + ", want " + std::to_string(want) + ")");
}

// ── the two configurations, as ABI facts ────────────────────────────────────

pie_forward::PieForwardLlamaLikeFacts qwen3_unfused_facts() {
    pie_forward::PieForwardLlamaLikeFacts f{};
    f.hidden = 1024;
    f.layers = 28;
    f.q_heads = 16;
    f.kv_heads = 8;
    f.head_dim = 128;
    f.intermediate = 3072;
    f.vocab = 151936;
    f.rope = static_cast<std::uint32_t>(pie_forward::PieForwardRopeKind::Standard);
    f.norm_variant = static_cast<std::uint32_t>(pie_forward::PieForwardNormVariant::Plain);
    f.qk_norm = 1;
    f.fused_qkv = 0;  // the golden this test feeds is the unfused trace
    f.tied_embeddings = 1;
    return f;
}

pie_forward::PieForwardLlamaLikeFacts phi3_facts() {
    pie_forward::PieForwardLlamaLikeFacts f{};
    f.hidden = 3072;
    f.layers = 32;
    f.q_heads = 32;
    f.kv_heads = 32;
    f.head_dim = 96;
    f.intermediate = 8192;
    f.vocab = 32064;
    f.rope = static_cast<std::uint32_t>(pie_forward::PieForwardRopeKind::Standard);
    f.norm_variant = static_cast<std::uint32_t>(pie_forward::PieForwardNormVariant::Plain);
    f.qk_norm = 0;
    f.fused_qkv = 0;
    f.tied_embeddings = 0;
    return f;
}

LlamaLikeGeometry geometry_of(const pie_forward::PieForwardLlamaLikeFacts& f) {
    return LlamaLikeGeometry{
        static_cast<int>(f.hidden),       static_cast<int>(f.layers),
        static_cast<int>(f.q_heads),      static_cast<int>(f.kv_heads),
        static_cast<int>(f.head_dim),     static_cast<int>(f.intermediate),
        static_cast<int>(f.vocab)};
}

// ── golden cross-check: the ABI plan is the JSON artifact, op for op ────────

const char* abi_kind_name(pie_forward::PieForwardOpKind k) {
    using K = pie_forward::PieForwardOpKind;
    switch (k) {
        case K::Embed: return "Embed";
        case K::Matmul: return "Matmul";
        case K::Rmsnorm: return "Rmsnorm";
        case K::RmsnormPerHead: return "RmsnormPerHead";
        case K::SplitQkv: return "SplitQkv";
        case K::Rope: return "Rope";
        case K::KvAppend: return "KvAppend";
        case K::Attention: return "Attention";
        case K::Swiglu: return "Swiglu";
        case K::LmHead: return "LmHead";
    }
    return "?";
}

json load_golden(const std::string& file) {
    const std::string path = std::string(PIE_FORWARD_GOLDEN_DIR) + "/" + file;
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open golden " + path);
    return json::parse(in);
}

void abi_plan_matches_golden(const pie_forward::ForwardPlan& plan, const std::string& file) {
    std::printf("[abi == %s]\n", file.c_str());
    const json golden = load_golden(file);
    const auto& ops = golden.at("ops");

    expect(golden.at("family") == "llama_like" &&
               plan.family() == "llama_like",
           "both forms name the family");
    if (!expect_eq(static_cast<long long>(plan.op_count()),
                   static_cast<long long>(ops.size()), "op counts agree")) {
        return;
    }

    bool kinds = true, weights = true, layers = true, operands = true;
    for (std::size_t i = 0; i < ops.size(); ++i) {
        const auto& gop = ops[i];
        const auto& op = plan.op(i);

        // `kind` serializes as {"Matmul": {payload...}}.
        const auto& kind_obj = gop.at("kind");
        const std::string gkind = kind_obj.begin().key();
        kinds = kinds && gkind == abi_kind_name(op.kind);

        const auto& payload = kind_obj.begin().value();
        const std::string gweight =
            payload.contains("weight") ? payload.at("weight").get<std::string>() : "";
        weights = weights && gweight == std::string(plan.weight_name(op));

        const int glayer = gop.at("layer").is_null() ? -1 : gop.at("layer").get<int>();
        layers = layers && glayer == static_cast<int>(op.layer);

        const auto in = plan.inputs(op);
        const auto out = plan.outputs(op);
        const auto& gin = gop.at("inputs");
        const auto& gout = gop.at("outputs");
        bool same = gin.size() == in.size && gout.size() == out.size;
        for (std::size_t j = 0; same && j < in.size; ++j) {
            same = gin[j].get<std::uint32_t>() == in[j];
        }
        for (std::size_t j = 0; same && j < out.size; ++j) {
            same = gout[j].get<std::uint32_t>() == out[j];
        }
        operands = operands && same;
    }
    expect(kinds, "every op kind matches the golden");
    expect(weights, "every weight name matches the golden");
    expect(layers, "every layer bracket matches the golden");
    expect(operands, "every op's SSA operands match the golden");
}

// ── the emitted DAG's shape ─────────────────────────────────────────────────

std::vector<Kind> layer_body(bool qk_norm) {
    std::vector<Kind> body{Kind::RmsNorm, Kind::ProjQ, Kind::ProjK, Kind::ProjV};
    if (qk_norm) {
        body.push_back(Kind::RmsNormPerHead);
        body.push_back(Kind::RmsNormPerHead);
    }
    for (Kind k : {Kind::Rope, Kind::KvAppend, Kind::Attention, Kind::ProjO,
                   Kind::RmsNorm, Kind::MlpGateUp, Kind::Swiglu, Kind::MlpDown}) {
        body.push_back(k);
    }
    return body;
}

void dag_has_the_declared_shape(const std::vector<Dispatch>& dag,
                                const LlamaLikeGeometry& g, bool qk_norm,
                                const std::string& tag) {
    std::printf("[dag shape: %s]\n", tag.c_str());
    const std::vector<Kind> body = layer_body(qk_norm);

    // Total: prologue + every layer's body + final norm + lm head. One
    // descriptor per plan op — v0 declares the trace verbatim.
    expect_eq(static_cast<long long>(dag.size()),
              1 + static_cast<long long>(g.layers) * static_cast<long long>(body.size()) + 2,
              "total dispatch count");

    bool dense = true;
    for (std::size_t i = 0; i < dag.size(); ++i) {
        dense = dense && dag[i].ordinal == static_cast<int>(i);
    }
    expect(dense, "ordinals are a dense 0..N-1 run");

    expect(!dag.empty() && dag.front().kind == Kind::Embed && dag.front().layer == -1,
           "the prologue is one layer-less embed");
    expect(dag.size() >= 2 && dag[dag.size() - 2].kind == Kind::FinalNorm &&
               dag[dag.size() - 2].layer == -1 && dag.back().kind == Kind::LmHead &&
               dag.back().layer == -1,
           "the epilogue is final norm then lm head, layer-less");

    // Per-layer kind sequence, every layer.
    bool sequence = true, bracketed = true;
    std::size_t at = 1;
    for (int L = 0; L < g.layers; ++L) {
        for (Kind want : body) {
            if (at >= dag.size() || dag[at].kind != want) {
                sequence = false;
                break;
            }
            bracketed = bracketed && dag[at].layer == L;
            ++at;
        }
        if (!sequence) break;
    }
    expect(sequence, "every layer emits the declared kind sequence");
    expect(bracketed, "every layer dispatch carries its layer");

    // Residual fusion sits exactly where the trace's beta_one accumulates
    // sit: the o_proj and the down projection, and nowhere else.
    bool fused_right = true;
    int fused = 0;
    for (const Dispatch& d : dag) {
        if (d.fuse_residual) {
            ++fused;
            fused_right = fused_right &&
                          (d.kind == Kind::ProjO || d.kind == Kind::MlpDown);
        } else {
            fused_right = fused_right &&
                          !(d.kind == Kind::ProjO || d.kind == Kind::MlpDown);
        }
    }
    expect(fused_right, "fuse_residual is exactly the o_proj/down accumulates");
    expect_eq(fused, 2LL * g.layers, "two fused-residual projections per layer");
}

// ── hazard-free coloring over the plan's SSA dataflow ───────────────────────
//
// The descriptors carry the plan's SSA value ids, so the shared live-range
// colourer generalizes for free: one Use per read/write, and — v0 declares no
// concurrency runs — run_ends is the identity, meaning a barrier after every
// dispatch. Hazard-freedom then says the pool assignment respects the SSA
// dataflow, which is the property the Mac increment's binder will lean on.
void coloring_is_hazard_free(const pie_forward::ForwardPlan& plan,
                             const std::vector<Dispatch>& dag, const std::string& tag) {
    std::printf("[scratch coloring: %s]\n", tag.c_str());

    std::vector<pie::metal::scratch::Use> uses;
    for (const Dispatch& d : dag) {
        for (int i = 0; i < d.n_reads; ++i) {
            uses.push_back({d.ordinal, 0, static_cast<int>(d.reads[i]), false});
        }
        for (int i = 0; i < d.n_writes; ++i) {
            uses.push_back({d.ordinal, 0, static_cast<int>(d.writes[i]), true});
        }
    }
    std::vector<int> run_ends(dag.size());
    for (std::size_t i = 0; i < dag.size(); ++i) run_ends[i] = static_cast<int>(i);

    const int value_count = static_cast<int>(plan.value_count());
    const auto c = pie::metal::scratch::color_live_ranges(
        uses, run_ends, value_count, /*no_recycle=*/false);

    expect(c.hazard_free, "no two overlapping values share a buffer");
    expect(c.colors_used > 0 && c.colors_used < 16,
           "the pool is a handful of buffers, not one per value (" +
               std::to_string(c.colors_used) + ")");

    // Recycling is the point: hundreds of SSA values must not become
    // hundreds of buffers.
    const auto nr = pie::metal::scratch::color_live_ranges(
        uses, run_ends, value_count, /*no_recycle=*/true);
    expect_eq(nr.colors_used, value_count, "no_recycle gives every value its own buffer");
    expect(c.colors_used < nr.colors_used / 10, "recycling collapses that by >10x");

    // Read-before-write over the walk order: the plan is SSA and the emitted
    // order preserves it, so nothing may read a value not yet written.
    std::vector<bool> written(static_cast<std::size_t>(value_count), false);
    int early = 0;
    for (const auto& u : uses) {
        if (u.is_write) {
            written[static_cast<std::size_t>(u.value)] = true;
        } else if (!written[static_cast<std::size_t>(u.value)]) {
            ++early;
        }
    }
    expect_eq(early, 0, "no dispatch reads a value nothing has written");
}

// ── the loud refusals ───────────────────────────────────────────────────────

void a_fused_trace_is_refused() {
    std::printf("[fused trace]\n");
    pie_forward::PieForwardLlamaLikeFacts f = qwen3_unfused_facts();
    f.fused_qkv = 1;
    const pie_forward::ForwardPlan plan = pie_forward::ForwardPlan::trace_llama_like(f);
    bool threw = false;
    std::string what;
    try {
        build_llama_like_declared_dag(plan, geometry_of(f));
    } catch (const std::exception& e) {
        threw = true;
        what = e.what();
    }
    expect(threw, "a fused-QKV trace throws rather than half-emitting");
    expect(what.find("UNFUSED") != std::string::npos ||
               what.find("fused") != std::string::npos,
           "and the message says why: " + what);
}

// The first `dyn` traced form — the qwen3_5_moe MLP-block fragment, whose
// TopK / selector-carrying Matmuls / WeightedSum the emitters do not
// consume — must be refused loudly, never half-emitted. The builder
// refuses at the first MoE-only weight name (`layer.0.router`), before any
// dyn op kind is even reached; the switch's default arm ("has no emission
// rule") backs that up for any kind past the v0 vocabulary. Either way:
// a throw, with a message naming the reason.
void a_dyn_trace_is_refused() {
    std::printf("[dyn (moe) trace]\n");
    pie_forward::PieForwardQwen35MoeMlpFacts f{};
    f.hidden = 2048;
    f.num_experts = 256;
    f.top_k = 8;
    f.moe_intermediate = 512;
    f.shared_expert_intermediate = 512;
    // Plain, so the refusal fires on the MoE-only weight name rather than
    // on the (also-refused) Gemma norm variant that qwen3.5 really uses —
    // this test is about the dyn vocabulary, not the norm fold.
    f.norm_variant = static_cast<std::uint32_t>(pie_forward::PieForwardNormVariant::Plain);
    const pie_forward::ForwardPlan plan = pie_forward::ForwardPlan::trace_qwen3_5_moe_mlp(f);
    expect(plan.op_count() == 13, "the ABI hands over the 13-op MoE fragment");

    LlamaLikeGeometry g{};
    g.layers = 1;
    g.hidden = 2048;
    bool threw = false;
    std::string what;
    try {
        build_llama_like_declared_dag(plan, g);
    } catch (const std::exception& e) {
        threw = true;
        what = e.what();
    }
    expect(threw, "a dyn (MoE) trace throws rather than half-emitting");
    expect(what.find("unknown weight") != std::string::npos ||
               what.find("no emission rule") != std::string::npos,
           "and the message says why: " + what);
}

// The GDN (gated-deltanet) fragment — SplitGdn / CausalConv1d / GdnPrep /
// GatedDelta / RmsnormGated, the first ops addressing PER-REQUEST state —
// must be refused the same way. As with the MoE fragment, the builder
// refuses at the first GDN-only weight name (`layer.0.in_proj_qkv`, op 2 of
// the unfused trace) before any GDN op kind is reached; the switch's
// default arm ("has no emission rule") backs that up for the kinds
// themselves. Either way: a throw, with a message naming the reason.
void a_gdn_trace_is_refused() {
    std::printf("[gdn (per-request state) trace]\n");
    pie_forward::PieForwardQwen35GdnFacts f{};
    // Qwen3.5-0.8B dims (crates/driver-metal/csrc/src/model/qwen3_5/geometry.hpp),
    // matching the qwen3_5_gdn_0_8b golden except the norm variant:
    // Plain, so the refusal fires on the GDN-only weight name rather than
    // on the (also-refused) Gemma fold qwen3.5 really uses — this test is
    // about the GDN vocabulary, not the norm fold.
    f.hidden = 1024;
    f.key_heads = 16;
    f.value_heads = 16;
    f.key_head_dim = 128;
    f.value_head_dim = 128;
    f.conv_kernel = 4;
    f.fused_in_proj = 0;  // the default binding (PIE_QWEN35_FUSED_GDN_PROJ unset)
    f.norm_variant = static_cast<std::uint32_t>(pie_forward::PieForwardNormVariant::Plain);
    const pie_forward::ForwardPlan plan = pie_forward::ForwardPlan::trace_qwen3_5_gdn(f);
    expect(plan.op_count() == 10, "the ABI hands over the 10-op GDN fragment");

    LlamaLikeGeometry g{};
    g.layers = 1;
    g.hidden = 1024;
    bool threw = false;
    std::string what;
    try {
        build_llama_like_declared_dag(plan, g);
    } catch (const std::exception& e) {
        threw = true;
        what = e.what();
    }
    expect(threw, "a GDN (per-request state) trace throws rather than half-emitting");
    expect(what.find("unknown weight") != std::string::npos ||
               what.find("no emission rule") != std::string::npos,
           "and the message says why: " + what);
}

// The full-attention fragment — qwen3.5's gated attention: SplitQGate /
// SigmoidGateMul plus partial Rope and the Gemma per-head fold riding as
// appended params — must be refused the same way. This one exercises the
// switch's DEFAULT arm directly: with a Plain norm variant the walk gets
// past the block norm and the three projections (q_proj/k_proj/v_proj are
// llama_like names too, so no weight-name refusal fires first) and lands on
// SplitQGate, an op kind past the v0 vocabulary — "has no emission rule".
// That is the appended-kind discipline pinned end to end: kinds 19/20 cross
// the ABI and cannot half-emit.
void a_full_attn_trace_is_refused() {
    std::printf("[full-attn (gated attention) trace]\n");
    pie_forward::PieForwardQwen35FullAttnFacts f{};
    // Qwen3.5-0.8B full-attention dims
    // (crates/driver-metal/csrc/src/model/qwen3_5/geometry.hpp), matching the
    // qwen3_5_full_attn_0_8b golden except the norm variant: Plain, so the
    // refusal fires on the new op KIND rather than on the (also-refused)
    // Gemma fold qwen3.5 really uses — this test is about the appended
    // kinds, not the norm fold.
    f.hidden = 1024;
    f.q_heads = 8;
    f.kv_heads = 2;
    f.head_dim = 256;
    f.rotary_dim = 64;  // partial_rotary_factor 0.25 x head_dim 256
    f.fused_qkv = 0;    // the default binding (PIE_QWEN35_FUSED_FULL_ATTN_QGKV unset)
    f.norm_variant = static_cast<std::uint32_t>(pie_forward::PieForwardNormVariant::Plain);
    const pie_forward::ForwardPlan plan = pie_forward::ForwardPlan::trace_qwen3_5_full_attn(f);
    expect(plan.op_count() == 12, "the ABI hands over the 12-op full-attn fragment");

    LlamaLikeGeometry g{};
    g.layers = 1;
    g.hidden = 1024;
    g.head_dim = 256;  // matches the trace, so no earlier check can mask the kind
    bool threw = false;
    std::string what;
    try {
        build_llama_like_declared_dag(plan, g);
    } catch (const std::exception& e) {
        threw = true;
        what = e.what();
    }
    expect(threw, "a gated-attention trace throws rather than half-emitting");
    expect(what.find("no emission rule") != std::string::npos,
           "and the default arm names the reason: " + what);
}

// The hybrid model composes all three fragment vocabularies; it must be
// refused before any GDN/full-attention op half-emits. Layer 0 of the 3:1
// schedule is a GDN layer, so (with the Plain-variant facts this test uses)
// the walk refuses at the first GDN-only weight name — the same first
// refusal the standalone GDN fragment hits — after accepting the embed
// prologue.
void a_hybrid_trace_is_refused() {
    std::printf("[hybrid (whole-model) trace]\n");
    pie_forward::PieForwardQwen35HybridFacts f{};
    // Qwen3.5-0.8B model dims (geometry.hpp): 24 layers, one full-attention
    // layer every 4th, dense MLP 3584, tied lm_head over vocab 248320.
    f.layers = 24;
    f.full_attn_interval = 4;
    f.vocab = 248320;
    f.tied_embeddings = 1;
    f.norm_variant = static_cast<std::uint32_t>(pie_forward::PieForwardNormVariant::Plain);
    f.attn.hidden = 1024;
    f.attn.q_heads = 8;
    f.attn.kv_heads = 2;
    f.attn.head_dim = 256;
    f.attn.rotary_dim = 64;
    f.attn.fused_qkv = 0;
    f.attn.norm_variant = static_cast<std::uint32_t>(pie_forward::PieForwardNormVariant::Plain);
    f.gdn.hidden = 1024;
    f.gdn.key_heads = 16;
    f.gdn.value_heads = 16;
    f.gdn.key_head_dim = 128;
    f.gdn.value_head_dim = 128;
    f.gdn.conv_kernel = 4;
    f.gdn.fused_in_proj = 0;
    f.gdn.norm_variant = static_cast<std::uint32_t>(pie_forward::PieForwardNormVariant::Plain);
    f.mlp_is_moe = 0;
    f.dense_intermediate = 3584;
    const pie_forward::ForwardPlan plan = pie_forward::ForwardPlan::trace_qwen3_5_hybrid(f);
    // 18 GDN layers x 14 + 6 full layers x 16 + embed/final norm/lm_head.
    expect(plan.op_count() == 18 * 14 + 6 * 16 + 3,
           "the ABI hands over the 351-op hybrid model");

    LlamaLikeGeometry g{};
    g.layers = 24;
    g.hidden = 1024;
    g.head_dim = 256;
    bool threw = false;
    std::string what;
    try {
        build_llama_like_declared_dag(plan, g);
    } catch (const std::exception& e) {
        threw = true;
        what = e.what();
    }
    expect(threw, "a hybrid trace throws rather than half-emitting");
    expect(what.find("unknown weight") != std::string::npos ||
               what.find("no emission rule") != std::string::npos,
           "and the message says why: " + what);
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("llama_like declared-dag emitter (backend #2 over pie_forward)\n");

    const auto qwen_facts = qwen3_unfused_facts();
    const pie_forward::ForwardPlan qwen =
        pie_forward::ForwardPlan::trace_llama_like(qwen_facts);
    abi_plan_matches_golden(qwen, "qwen3_0_6b_unfused_qkv.json");
    const auto qwen_dag = build_llama_like_declared_dag(qwen, geometry_of(qwen_facts));
    dag_has_the_declared_shape(qwen_dag, geometry_of(qwen_facts), /*qk_norm=*/true,
                               "qwen3-0.6B unfused, qk-norm on");
    coloring_is_hazard_free(qwen, qwen_dag, "qwen3-0.6B");

    const auto phi_facts = phi3_facts();
    const pie_forward::ForwardPlan phi =
        pie_forward::ForwardPlan::trace_llama_like(phi_facts);
    abi_plan_matches_golden(phi, "phi3_mini.json");
    const auto phi_dag = build_llama_like_declared_dag(phi, geometry_of(phi_facts));
    dag_has_the_declared_shape(phi_dag, geometry_of(phi_facts), /*qk_norm=*/false,
                               "phi3-mini unfused, qk-norm off");
    coloring_is_hazard_free(phi, phi_dag, "phi3-mini");

    a_fused_trace_is_refused();
    a_dyn_trace_is_refused();
    a_gdn_trace_is_refused();
    a_full_attn_trace_is_refused();
    a_hybrid_trace_is_refused();

    std::printf("\n==== llama_like_declared_dag_test: %s ====\n",
                g_failures == 0 ? "all passed" : "FAILURES");
    return g_failures == 0 ? 0 : 1;
}
